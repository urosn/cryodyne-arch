#!/usr/bin/env python3
"""CRYODYNE korak 0: memorijski otisak scenarija iz workloads.yaml.

Racuna (aproksimativno, iz dimenzija config.json):
  - broj parametara (ukupno i aktivno po tokenu),
  - velicinu tezina u izabranom formatu,
  - KV kes po tokenu i ukupno za kontekst, stanje rekurencije po sekvenci,
  - ukupnu memoriju scenarija i da li staje na karticu,
  - bajtove tezina procitane po dekodiranom tokenu (batch 1 => ogranicenje propusnim opsegom).
Pokretanje:  python3 footprint.py workloads.yaml
"""
import sys, csv, yaml

GB = 1e9


def attn_params(h, a):
    q = a["q_heads"] * a["head_dim"] * (2 if a.get("output_gate") else 1)
    kv = 2 * a["kv_heads"] * a["head_dim"]
    return h * (q + kv) + a["q_heads"] * a["head_dim"] * h


def mamba2_params(h, m):
    d_inner = m["heads"] * m["head_dim"]
    conv_dim = d_inner + 2 * m["groups"] * m["state"]
    in_proj = h * (2 * d_inner + 2 * m["groups"] * m["state"] + m["heads"])
    return in_proj + conv_dim * m["conv_kernel"] + d_inner * h + 2 * m["heads"] + d_inner


def gdn_params(h, g):
    kq = 2 * g["k_heads"] * g["k_dim"]
    v = g["v_heads"] * g["v_dim"]
    return h * (kq + 2 * v + 2 * g["v_heads"]) + (kq + v) * g["conv_kernel"] + v * h


def kda_params(h, k):
    d = k["heads"] * k["head_dim"]
    # q,k,v,izlazna kapija, kapija zaboravljanja (aproks. puni rang), beta, izlaz
    return h * (5 * d + k["heads"]) + 3 * d * k["conv_kernel"] + d * h


def mla_params(h, m):
    qd = m["heads"] * (m["qk_nope"] + m["qk_rope"])
    kv_a = h * (m["kv_lora_rank"] + m["qk_rope"])
    kv_b = m["kv_lora_rank"] * m["heads"] * (m["qk_nope"] + m["v_dim"])
    return h * qd + kv_a + kv_b + m["heads"] * m["v_dim"] * h


def mlp_params(h, inter, gated):
    return h * inter * (3 if gated else 2)


def model_stats(md):
    h = md["hidden"]
    L = md["layers"]
    body_total = body_active = 0
    kv_per_tok = 0        # broj vrednosti po tokenu
    state_per_seq = 0     # broj vrednosti po sekvenci
    if "attention" in L and "attention" in md:
        p = attn_params(h, md["attention"]) * L["attention"]
        body_total += p; body_active += p
        a = md["attention"]
        kv_per_tok += 2 * a["kv_heads"] * a["head_dim"] * L["attention"]
    if "mamba2" in L:
        m = md["mamba2"]
        p = mamba2_params(h, m) * L["mamba2"]
        body_total += p; body_active += p
        state_per_seq += m["heads"] * m["head_dim"] * m["state"] * L["mamba2"]
        state_per_seq += (m["heads"] * m["head_dim"] + 2 * m["groups"] * m["state"]) * (m["conv_kernel"] - 1) * L["mamba2"]
    if "gated_deltanet" in L:
        g = md["gated_deltanet"]
        p = gdn_params(h, g) * L["gated_deltanet"]
        body_total += p; body_active += p
        state_per_seq += g["v_heads"] * g["k_dim"] * g["v_dim"] * L["gated_deltanet"]
    if "kda" in L:
        k = md["kda"]
        p = kda_params(h, k) * L["kda"]
        body_total += p; body_active += p
        state_per_seq += k["heads"] * k["head_dim"] * k["head_dim"] * L["kda"]
    if "mla" in L:
        m = md["mla"]
        p = mla_params(h, m) * L["mla"]
        body_total += p; body_active += p
        kv_per_tok += (m["kv_lora_rank"] + m["qk_rope"]) * L["mla"]
    if "mlp" in L:
        mp = md["mlp"]
        p = mlp_params(h, mp["intermediate"], mp["gated"]) * L["mlp"]
        body_total += p; body_active += p
    if "moe" in md:
        mo = md["moe"]
        n_layers = sum(L.values())
        dense = mo.get("first_dense_layers", 0)
        moe_layers = n_layers - dense
        expert = mlp_params(h, mo["expert_intermediate"], True)
        shared = mlp_params(h, mo.get("shared_intermediate", mo["expert_intermediate"]), True) * mo.get("shared_experts", 1)
        router = h * mo["experts"]
        body_total += moe_layers * (mo["experts"] * expert + shared + router)
        body_active += moe_layers * (mo["active"] * expert + shared + router)
        if dense:
            d = mlp_params(h, mo["dense_intermediate"], True) * dense
            body_total += d; body_active += d
    emb = md["vocab"] * h * (1 if md.get("tied_embeddings") else 2)
    return dict(body_total=body_total, body_active=body_active, emb=emb,
                kv_per_tok=kv_per_tok, state_per_seq=state_per_seq)


def main(path):
    cfg = yaml.safe_load(open(path))
    fmts, models = cfg["formats"], cfg["models"]
    eb, kvb, sb = cfg["embedding_bits"], cfg["kv_bits"], cfg["state_bits"]
    rows = []
    print("\nMODELI (izracunato, aproksimativno)")
    print(f"{'model':22} {'ukupno [B]':>11} {'aktivno [B]':>12} {'KV/token [kB]':>14} {'stanje/sekv [MB]':>17}")
    for name, md in models.items():
        s = model_stats(md)
        print(f"{name:22} {(s['body_total']+s['emb'])/1e9:11.1f} {(s['body_active']+s['emb'])/1e9:12.2f} "
              f"{s['kv_per_tok']*kvb/8/1e3:14.1f} {s['state_per_seq']*sb/8/1e6:17.1f}")
    print("\nSCENARIJI")
    hdr = ["id", "segment", "model", "format", "batch", "context", "weights_GB", "kv_GB", "state_GB",
           "total_GB", "card_GB", "fits", "weight_bytes_per_token_GB", "bytes_per_step_full_ctx_GB", "tok_s_upper_bound_at_500GBs"]
    print(" ".join(f"{x:>10}" for x in hdr))
    for sc in cfg["scenarios"]:
        md = models[sc["model"]]; s = model_stats(md)
        bits = fmts[sc["format"]]
        w = (s["body_total"] * bits + s["emb"] * eb) / 8 / GB
        kv = s["kv_per_tok"] * sc["context"] * sc["batch"] * kvb / 8 / GB
        st = s["state_per_seq"] * sc["batch"] * sb / 8 / GB
        tot = w + kv + st
        card = cfg["cards"][sc["segment"]]["memory_gb"]
        # po dekodiranom koraku procita se aktivni deo tela + lm_head (pola embeddinga ako nisu vezani)
        lm_head = md["vocab"] * md["hidden"]
        wpt = (s["body_active"] * bits + lm_head * eb) / 8 / GB
        # jedan korak dekodiranja (ceo batch) na punom kontekstu: tezine jednom + ceo KV + stanje citanje i upis
        step = wpt + kv + 2 * st
        tps = 500 / step * sc["batch"]   # gornja granica: kartica 500 GB/s, samo memorijski saobracaj (procena)
        r = [sc["id"], sc["segment"], sc["model"], sc["format"], sc["batch"], sc["context"],
             round(w, 1), round(kv, 1), round(st, 2), round(tot, 1), card, "DA" if tot <= card * 0.9 else "NE", round(wpt, 2),
             round(step, 1), round(tps)]
        rows.append(r)
        print(" ".join(f"{str(x):>10}" for x in r))
    with open("footprint.csv", "w", newline="") as f:
        csv.writer(f).writerows([hdr] + rows)
    print("\n'fits' = DA ako ukupno <= 90% memorije kartice (10% rezerva za aktivacije i runtime; procena).")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "workloads.yaml")
