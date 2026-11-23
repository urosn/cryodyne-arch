#!/usr/bin/env python3
"""CRYODYNE korak 1: merenje pojedinacnih slojeva na stvarnim dimenzijama modela.

Ideja: slojevi se u modelu ponavljaju, pa nije potrebno preuzimati stotine GB tezina.
Za svaki tip sloja svakog modela iz workloads.yaml meri se:
  - "gemm":  linearne projekcije sloja (oblici matrica tacno kao u modelu),
  - "core":  jezgro sloja (atencija nad KV kesom, Mamba-2 sken, gated delta pravilo, MLA, MoE),
  za prefill (T tokena) i za jedan korak dekodiranja (batch B, kontekst C).
Tezine su slucajne - brzina ne zavisi od vrednosti tezina.

Pokretanje (GPU):  python3 bench_layers.py --workloads ../step0/workloads.yaml --out results
Test na CPU-u:      python3 bench_layers.py --smoke
"""
import argparse, csv, json, os, platform, sys, time, traceback
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import layers_ref as R

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.bfloat16 if DEV == "cuda" else torch.float32

# -------- opcione optimizovane biblioteke --------
LIBS = {}
try:
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
    LIBS["mamba_ssm"] = True
except Exception as e:
    LIBS["mamba_ssm"] = f"nedostupno: {e.__class__.__name__}"
try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
    LIBS["fla_gdn"] = True
except Exception as e:
    LIBS["fla_gdn"] = f"nedostupno: {e.__class__.__name__}"
try:
    from fla.ops.kda import chunk_kda, fused_recurrent_kda  # postoji u novijim verzijama fla
    LIBS["fla_kda"] = True
except Exception as e:
    LIBS["fla_kda"] = f"nedostupno: {e.__class__.__name__}"


def timeit(fn, warmup=3, iters=10):
    """Prosecno vreme u ms. Na GPU-u CUDA eventi; na CPU-u wall clock."""
    for _ in range(warmup):
        fn()
    if DEV == "cuda":
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record(); torch.cuda.synchronize()
        return s.elapsed_time(e) / iters
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t) * 1000 / iters


def rnd(*shape, dtype=None):
    return torch.randn(*shape, device=DEV, dtype=dtype or DT)


def gemm_case(rows, in_f, out_list):
    """Vreme svih projekcija sloja kao niz GEMM-ova."""
    x = rnd(rows, in_f)
    ws = [rnd(o, i) for i, o in out_list]
    xs = {in_f: x}
    for i, _ in out_list:
        xs.setdefault(i, rnd(rows, i))
    return timeit(lambda: [F.linear(xs[i], w) for (i, _), w in zip(out_list, ws)])


# ---------------- jezgra slojeva ----------------
def core_attention(a, B, ctx, T):
    hq, hkv, d = a["q_heads"], a["kv_heads"], a["head_dim"]
    if T == 1:   # dekodiranje: 1 upit nad KV kesom duzine ctx
        q, k, v = rnd(B, hq, 1, d), rnd(B, hkv, ctx, d), rnd(B, hkv, ctx, d)
        return timeit(lambda: F.scaled_dot_product_attention(q, k, v, enable_gqa=True)), "sdpa"
    q, k, v = rnd(B, hq, T, d), rnd(B, hkv, T, d), rnd(B, hkv, T, d)
    return timeit(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)), "sdpa"


def core_mamba2(m, B, T):
    h, p, n, g = m["heads"], m["head_dim"], m["state"], m["groups"]
    if LIBS["mamba_ssm"] is True and DEV == "cuda":
        A = -torch.rand(h, device=DEV, dtype=torch.float32)
        if T == 1:
            st = torch.zeros(B, h, p, n, device=DEV, dtype=DT)
            x, dt = rnd(B, h, p), torch.rand(B, h, device=DEV, dtype=DT)[..., None].expand(B, h, p)
            Ae = A[:, None, None].expand(h, p, n)
            Bm, Cm = rnd(B, g, n), rnd(B, g, n)
            return timeit(lambda: selective_state_update(st, x, dt, Ae, Bm, Cm)), "mamba_ssm"
        x, dt = rnd(B, T, h, p), torch.rand(B, T, h, device=DEV, dtype=DT)
        Bm, Cm = rnd(B, T, g, n), rnd(B, T, g, n)
        return timeit(lambda: mamba_chunk_scan_combined(x, dt, A, Bm, Cm, chunk_size=256)), "mamba_ssm"
    if T > 64 and DEV == "cuda":
        return None, "preskoceno (nema mamba_ssm)"
    layer = R.Mamba2Ref(m.get("_hidden", 64), h, p, n, g).to(DEV, DT)
    x = rnd(B, T, m.get("_hidden", 64))
    with torch.no_grad():
        return timeit(lambda: layer(x), 1, 2), "ref"


def core_delta(cfg, B, T, kda):
    hk = cfg.get("k_heads", cfg.get("heads")); hv = cfg.get("v_heads", cfg.get("heads"))
    dk = cfg.get("k_dim", cfg.get("head_dim")); dv = cfg.get("v_dim", cfg.get("head_dim"))
    lib = "fla_kda" if kda else "fla_gdn"
    if LIBS[lib] is True and DEV == "cuda":
        q, k, v = rnd(B, T, hv, dk), F.normalize(rnd(B, T, hv, dk), dim=-1), rnd(B, T, hv, dv)
        beta = torch.rand(B, T, hv, device=DEV, dtype=DT)
        g = -torch.rand(B, T, hv, dk, device=DEV, dtype=torch.float32) if kda else -torch.rand(B, T, hv, device=DEV, dtype=torch.float32)
        s0 = torch.zeros(B, hv, dk, dv, device=DEV, dtype=torch.float32)
        if kda:
            fn = (lambda: fused_recurrent_kda(q, k, v, g, beta, initial_state=s0)) if T == 1 else (lambda: chunk_kda(q, k, v, g, beta))
        else:
            fn = (lambda: fused_recurrent_gated_delta_rule(q, k, v, g, beta, initial_state=s0)) if T == 1 else (lambda: chunk_gated_delta_rule(q, k, v, g, beta))
        return timeit(fn), lib
    if T > 64 and DEV == "cuda":
        return None, f"preskoceno (nema {lib})"
    hid = cfg.get("_hidden", 64)
    layer = R.DeltaRef(hid, hk, hv, dk, dv, per_channel_decay=kda).to(DEV, DT)
    x = rnd(B, T, hid)
    with torch.no_grad():
        return timeit(lambda: layer(x), 1, 2), "ref"


def core_mla_decode(m, B, ctx):
    """Apsorbovani oblik: upit se poredi direktno sa latentnim kesom [B, ctx, r+rope]."""
    h, r, dr = m["heads"], m["kv_lora_rank"], m["qk_rope"]
    q, cache = rnd(B, h, r + dr), rnd(B, ctx, r + dr)
    scale = 1.0 / (m["qk_nope"] + dr) ** 0.5

    def f():
        s = torch.softmax(torch.einsum("bhr,bcr->bhc", q, cache) * scale, -1)
        return torch.einsum("bhc,bcr->bhr", s, cache[..., :r])
    return timeit(f), "torch_absorbed"


def core_moe(mo, hidden, rows):
    """Aktivni eksperti za 'rows' tokena; batch 1 => mo['active'] eksperata."""
    E, k, I = mo["experts"], mo["active"], mo["expert_intermediate"]
    n_used = min(E, k * rows)
    w1, w2 = rnd(n_used, hidden, 2 * I), rnd(n_used, I, hidden)
    x = rnd(n_used, max(1, rows * k // n_used), hidden)

    def f():
        a, b = torch.bmm(x, w1).chunk(2, -1)
        return torch.bmm(F.silu(a) * b, w2)
    return timeit(f), f"bmm_{n_used}_eksperata"


# ---------------- oblici projekcija po tipu sloja ----------------
def proj_shapes(kind, md):
    h = md["hidden"]
    if kind == "attention":
        a = md["attention"]; q = a["q_heads"] * a["head_dim"] * (2 if a.get("output_gate") else 1)
        return [(h, q), (h, 2 * a["kv_heads"] * a["head_dim"]), (a["q_heads"] * a["head_dim"], h)]
    if kind == "mamba2":
        m = md["mamba2"]; di = m["heads"] * m["head_dim"]
        return [(h, 2 * di + 2 * m["groups"] * m["state"] + m["heads"]), (di, h)]
    if kind == "gated_deltanet":
        g = md["gated_deltanet"]; v = g["v_heads"] * g["v_dim"]
        return [(h, 2 * g["k_heads"] * g["k_dim"]), (h, v), (h, v), (h, 2 * g["v_heads"]), (v, h)]
    if kind == "kda":
        k = md["kda"]; d = k["heads"] * k["head_dim"]
        return [(h, 3 * d), (h, d), (h, d), (h, k["heads"]), (d, h)]
    if kind == "mla":
        m = md["mla"]
        return [(h, m["heads"] * (m["qk_nope"] + m["qk_rope"])), (h, m["kv_lora_rank"] + m["qk_rope"]),
                (m["kv_lora_rank"], m["heads"] * (m["qk_nope"] + m["v_dim"])), (m["heads"] * m["v_dim"], h)]
    if kind == "mlp":
        mp = md["mlp"]; n = 3 if mp["gated"] else 2
        return [(h, mp["intermediate"])] * (n - 1) + [(mp["intermediate"], h)]
    return []


def run(cfg, prefill_T, decode_cases, out_dir, smoke=False):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for name, md in cfg["models"].items():
        if smoke and name != "nemotron_h_8b" and name != "qwen3_next_80b_a3b" and name != "kimi_linear_48b_a3b":
            continue
        kinds = [k for k in md["layers"] if md["layers"][k] > 0]
        if "moe" in md:
            kinds.append("moe")
        for kind in kinds:
            phases = [("prefill", 1, prefill_T, prefill_T)] + [("decode", b, c, 1) for b, c in decode_cases]
            for phase, B, ctx, T in phases:
                rec = dict(model=name, layer=kind, count=md["layers"].get(kind, sum(md["layers"].values())),
                           phase=phase, batch=B, context=ctx, tokens=T)
                try:
                    shapes = proj_shapes(kind, md)
                    rec["gemm_ms"] = round(gemm_case(B * T, shapes[0][0], shapes), 4) if shapes else 0.0
                    if kind == "attention":
                        ms, impl = core_attention(md["attention"], B, ctx, T)
                    elif kind == "mamba2":
                        m = dict(md["mamba2"], _hidden=md["hidden"])
                        ms, impl = core_mamba2(m, B, T)
                    elif kind in ("gated_deltanet", "kda"):
                        c = dict(md[kind], _hidden=md["hidden"])
                        ms, impl = core_delta(c, B, T, kind == "kda")
                    elif kind == "mla":
                        ms, impl = core_mla_decode(md["mla"], B, ctx) if T == 1 else core_attention(
                            {"q_heads": md["mla"]["heads"], "kv_heads": md["mla"]["heads"],
                             "head_dim": md["mla"]["qk_nope"] + md["mla"]["qk_rope"]}, B, ctx, T)
                    elif kind == "moe":
                        rec["gemm_ms"] = 0.0
                        ms, impl = core_moe(md["moe"], md["hidden"], B * T)
                    else:   # mlp: sve je u GEMM-u
                        ms, impl = 0.0, "gemm"
                    rec["core_ms"] = None if ms is None else round(ms, 4)
                    rec["impl"] = impl
                except Exception as e:
                    rec["error"] = f"{e.__class__.__name__}: {str(e)[:200]}"
                    traceback.print_exc()
                if DEV == "cuda":
                    rec["peak_mem_GB"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
                    torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
                rows.append(rec)
                print(json.dumps(rec, ensure_ascii=False), flush=True)
    keys = ["model", "layer", "count", "phase", "batch", "context", "tokens", "gemm_ms", "core_ms", "impl", "peak_mem_GB", "error"]
    with open(os.path.join(out_dir, "layers.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, keys); w.writeheader(); [w.writerow({k: r.get(k) for k in keys}) for r in rows]
    env = dict(device=DEV, gpu=torch.cuda.get_device_name(0) if DEV == "cuda" else platform.processor(),
               torch=torch.__version__, cuda=torch.version.cuda, libs=LIBS, time=time.strftime("%Y-%m-%d %H:%M:%S"))
    json.dump(env, open(os.path.join(out_dir, "environment.json"), "w"), indent=2, ensure_ascii=False)
    print("\nOkruzenje:", json.dumps(env, ensure_ascii=False))


def shrink(cfg, f=16):
    """Smanjene dimenzije za test na CPU-u (proverava samo da kod radi)."""
    import copy
    c = copy.deepcopy(cfg)
    for md in c["models"].values():
        md["hidden"] = max(64, md["hidden"] // f)
        for key in ("attention", "mamba2", "gated_deltanet", "kda", "mla"):
            if key in md:
                for kk in ("heads", "q_heads", "kv_heads", "k_heads", "v_heads"):
                    if kk in md[key]:
                        md[key][kk] = max(1, md[key][kk] // 8)
                for kk in ("head_dim", "k_dim", "v_dim", "state", "kv_lora_rank", "qk_nope", "qk_rope"):
                    if kk in md[key]:
                        md[key][kk] = max(4, md[key][kk] // 8)
                if "groups" in md[key]:
                    md[key]["groups"] = 1
        if "mlp" in md:
            md["mlp"]["intermediate"] //= f
        if "moe" in md:
            md["moe"]["experts"] //= 16; md["moe"]["expert_intermediate"] //= 8
    return c


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workloads", default=os.path.join(os.path.dirname(__file__), "..", "step0", "workloads.yaml"))
    ap.add_argument("--out", default="results")
    ap.add_argument("--prefill", type=int, default=4096)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.workloads))
    if a.smoke:
        run(shrink(cfg), 16, [(1, 64), (4, 64)], a.out, smoke=True)
    else:
        # parovi (batch, kontekst) za dekodiranje: individualni i data centar
        run(cfg, a.prefill, [(1, 32768), (1, 131072), (32, 8192), (64, 8192), (128, 4096)], a.out)
