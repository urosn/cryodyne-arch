"""Referentne (cisto PyTorch) implementacije slojeva iz skupa opterecenja.

Svrha: (1) zlatni model ponasanja za korake 6 i 9, (2) rezervni put kad optimizovana
biblioteka (mamba_ssm, fla) nije instalirana, (3) test na CPU-u sa malim dimenzijama.
Na GPU-u se brzina meri optimizovanim kernelima (bench_layers.py ih bira automatski);
vreme ovih referentnih verzija NIJE merilo hardvera.
"""
import torch
import torch.nn.functional as F


# ---------------- atencija (GQA), jedan korak dekodiranja i prefill ----------------
class GQA(torch.nn.Module):
    def __init__(self, hidden, q_heads, kv_heads, head_dim, output_gate=False):
        super().__init__()
        self.hq, self.hkv, self.d, self.gate = q_heads, kv_heads, head_dim, output_gate
        self.q = torch.nn.Linear(hidden, q_heads * head_dim * (2 if output_gate else 1), bias=False)
        self.kv = torch.nn.Linear(hidden, 2 * kv_heads * head_dim, bias=False)
        self.o = torch.nn.Linear(q_heads * head_dim, hidden, bias=False)

    def forward(self, x, cache=None):
        B, T, _ = x.shape
        q = self.q(x)
        if self.gate:
            q, g = q.chunk(2, dim=-1)
        q = q.view(B, T, self.hq, self.d).transpose(1, 2)
        k, v = self.kv(x).view(B, T, 2, self.hkv, self.d).unbind(2)
        k, v = k.transpose(1, 2), v.transpose(1, 2)
        if cache is not None:
            k = torch.cat([cache[0], k], 2); v = torch.cat([cache[1], v], 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=(cache is None and T > 1), enable_gqa=True)
        y = y.transpose(1, 2).reshape(B, T, -1)
        if self.gate:
            y = y * torch.sigmoid(g)
        return self.o(y), (k, v)


# ---------------- Mamba-2 (SSD), sekvencijalna rekurencija ----------------
class Mamba2Ref(torch.nn.Module):
    def __init__(self, hidden, heads, head_dim, state, groups):
        super().__init__()
        self.h, self.p, self.n, self.g = heads, head_dim, state, groups
        d_inner = heads * head_dim
        self.inp = torch.nn.Linear(hidden, 2 * d_inner + 2 * groups * state + heads, bias=False)
        self.A = torch.nn.Parameter(-torch.rand(heads))
        self.out = torch.nn.Linear(d_inner, hidden, bias=False)

    def forward(self, x, S=None):
        B, T, _ = x.shape
        z, xs, Bm, Cm, dt = torch.split(self.inp(x), [self.h * self.p, self.h * self.p, self.g * self.n, self.g * self.n, self.h], -1)
        xs = xs.view(B, T, self.h, self.p); dt = F.softplus(dt)
        rep = self.h // self.g
        Bm = Bm.view(B, T, self.g, self.n).repeat_interleave(rep, 2)
        Cm = Cm.view(B, T, self.g, self.n).repeat_interleave(rep, 2)
        if S is None:
            S = x.new_zeros(B, self.h, self.p, self.n, dtype=torch.float32)
        ys = []
        for t in range(T):                       # stanje S: [B, glave, head_dim, state]
            a = torch.exp(dt[:, t] * self.A).float()[..., None, None]
            S = S * a + (dt[:, t, :, None, None] * xs[:, t, :, :, None] * Bm[:, t, :, None, :]).float()
            ys.append(torch.einsum("bhpn,bhn->bhp", S, Cm[:, t].float()).to(x.dtype))
        y = torch.stack(ys, 1).reshape(B, T, -1) * F.silu(z)
        return self.out(y), S


# ---------------- Gated DeltaNet i KDA (pravilo gated delta) ----------------
class DeltaRef(torch.nn.Module):
    """per_channel_decay=False -> Gated DeltaNet (skalarno zaboravljanje po glavi)
       per_channel_decay=True  -> KDA (zaboravljanje po kanalu kljuca)."""
    def __init__(self, hidden, k_heads, v_heads, k_dim, v_dim, per_channel_decay=False):
        super().__init__()
        self.hk, self.hv, self.dk, self.dv, self.pc = k_heads, v_heads, k_dim, v_dim, per_channel_decay
        self.qk = torch.nn.Linear(hidden, 2 * k_heads * k_dim, bias=False)
        self.v = torch.nn.Linear(hidden, v_heads * v_dim, bias=False)
        self.decay = torch.nn.Linear(hidden, v_heads * (k_dim if per_channel_decay else 1), bias=False)
        self.beta = torch.nn.Linear(hidden, v_heads, bias=False)
        self.gate = torch.nn.Linear(hidden, v_heads * v_dim, bias=False)
        self.out = torch.nn.Linear(v_heads * v_dim, hidden, bias=False)

    def forward(self, x, S=None):
        B, T, _ = x.shape
        q, k = self.qk(x).view(B, T, 2, self.hk, self.dk).unbind(2)
        rep = self.hv // self.hk
        q = F.normalize(q.repeat_interleave(rep, 2), dim=-1).float()
        k = F.normalize(k.repeat_interleave(rep, 2), dim=-1).float()
        v = self.v(x).view(B, T, self.hv, self.dv).float()
        alpha = torch.sigmoid(self.decay(x)).view(B, T, self.hv, -1).float()   # [.., 1] ili [.., dk]
        beta = torch.sigmoid(self.beta(x)).float()
        if S is None:
            S = x.new_zeros(B, self.hv, self.dv, self.dk, dtype=torch.float32)  # stanje: dv x dk po glavi
        ys = []
        for t in range(T):
            S = S * alpha[:, t, :, None, :]                        # zaboravljanje
            kt, vt, bt = k[:, t], v[:, t], beta[:, t, :, None]
            err = torch.einsum("bhvk,bhk->bhv", S, kt) - vt        # delta: S k - v
            S = S - bt[..., None] * err[..., None] * kt[:, :, None, :]
            ys.append(torch.einsum("bhvk,bhk->bhv", S, q[:, t]))
        y = torch.stack(ys, 1).reshape(B, T, -1).to(x.dtype) * torch.sigmoid(self.gate(x))
        return self.out(y), S


# ---------------- MLA (latentni KV kes), naivno ----------------
class MLARef(torch.nn.Module):
    def __init__(self, hidden, heads, kv_lora_rank, qk_nope, qk_rope, v_dim):
        super().__init__()
        self.h, self.r, self.dn, self.dr, self.dv = heads, kv_lora_rank, qk_nope, qk_rope, v_dim
        self.q = torch.nn.Linear(hidden, heads * (qk_nope + qk_rope), bias=False)
        self.kv_a = torch.nn.Linear(hidden, kv_lora_rank + qk_rope, bias=False)
        self.kv_b = torch.nn.Linear(kv_lora_rank, heads * (qk_nope + v_dim), bias=False)
        self.o = torch.nn.Linear(heads * v_dim, hidden, bias=False)

    def forward(self, x, cache=None):          # cache: [B, T, r + rope] - jedino sto se cuva
        B, T, _ = x.shape
        lat = self.kv_a(x)
        if cache is not None:
            lat = torch.cat([cache, lat], 1)
        c, kr = lat.split([self.r, self.dr], -1)
        kn, v = self.kv_b(c).view(B, -1, self.h, self.dn + self.dv).split([self.dn, self.dv], -1)
        k = torch.cat([kn, kr[:, :, None, :].expand(-1, -1, self.h, -1)], -1).transpose(1, 2)
        q = self.q(x).view(B, T, self.h, -1).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v.transpose(1, 2), is_causal=(cache is None and T > 1))
        return self.o(y.transpose(1, 2).reshape(B, T, -1)), lat


# ---------------- MoE (top-k), naivno ----------------
class MoERef(torch.nn.Module):
    def __init__(self, hidden, experts, active, inter, shared_inter):
        super().__init__()
        self.k = active
        self.router = torch.nn.Linear(hidden, experts, bias=False)
        self.w1 = torch.nn.Parameter(torch.randn(experts, hidden, 2 * inter) * 0.02)
        self.w2 = torch.nn.Parameter(torch.randn(experts, inter, hidden) * 0.02)
        self.shared = torch.nn.Sequential(torch.nn.Linear(hidden, shared_inter, bias=False), torch.nn.SiLU(),
                                          torch.nn.Linear(shared_inter, hidden, bias=False))

    def forward(self, x):
        B, T, H = x.shape
        xf = x.reshape(-1, H)
        w, idx = torch.topk(torch.softmax(self.router(xf).float(), -1), self.k, -1)
        out = torch.zeros_like(xf)
        for e in idx.unique().tolist():
            rows, slot = (idx == e).nonzero(as_tuple=True)
            a, b = (xf[rows] @ self.w1[e]).chunk(2, -1)
            out.index_add_(0, rows, ((F.silu(a) * b) @ self.w2[e]) * w[rows, slot, None].to(x.dtype))
        return (out + self.shared(xf)).view(B, T, H), idx.unique().numel()
