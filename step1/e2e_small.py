#!/usr/bin/env python3
"""CRYODYNE korak 1b: merenje celog malog modela i snimanje traga izvrsavanja.

Meri Nemotron-H 8B (BF16, staje na jedan GPU od 80 GB) od kraja do kraja:
  - vreme prefill-a za 4096 tokena, brzinu dekodiranja za batch 1,
  - PyTorch execution trace + Kineto trag (ulaz za Chakra u koraku 7),
  - tabelu kernela po vremenu (torch.profiler).
Sluzi da se proveri da zbir slojeva iz bench_layers.py odgovara celom modelu.

Pokretanje:  python3 e2e_small.py --model nvidia/Nemotron-H-8B-Base-8K --out results
"""
import argparse, json, os, time
import torch
from torch.profiler import profile, ProfilerActivity, ExecutionTraceObserver
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="nvidia/Nemotron-H-8B-Base-8K")
ap.add_argument("--prompt_len", type=int, default=4096)
ap.add_argument("--new_tokens", type=int, default=128)
ap.add_argument("--out", default="results")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
tag = a.model.split("/")[-1]

tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
ids = torch.randint(100, tok.vocab_size - 100, (1, a.prompt_len), device="cuda")

with torch.no_grad():
    model.generate(ids[:, :256], max_new_tokens=8, do_sample=False)          # zagrevanje
    torch.cuda.synchronize(); t0 = time.time()
    model(ids)                                                              # prefill
    torch.cuda.synchronize(); t_prefill = time.time() - t0
    t0 = time.time()
    out = model.generate(ids, max_new_tokens=a.new_tokens, do_sample=False, min_new_tokens=a.new_tokens)
    torch.cuda.synchronize(); t_total = time.time() - t0
    t_decode = max(1e-9, t_total - t_prefill)

    et = ExecutionTraceObserver()
    et.register_callback(os.path.join(a.out, f"{tag}_pytorch_et.json"))
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True,
                 execution_trace_observer=et) as prof:
        model.generate(ids[:, :1024], max_new_tokens=16, do_sample=False, min_new_tokens=16)
    prof.export_chrome_trace(os.path.join(a.out, f"{tag}_kineto.json"))
    with open(os.path.join(a.out, f"{tag}_kernels.txt"), "w") as f:
        f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=60))

res = dict(model=a.model, prompt_len=a.prompt_len, new_tokens=a.new_tokens,
           prefill_s=round(t_prefill, 3), decode_tok_s=round(a.new_tokens / t_decode, 2),
           peak_mem_GB=round(torch.cuda.max_memory_allocated() / 1e9, 2), gpu=torch.cuda.get_device_name(0),
           note="HF transformers u eager rezimu; sporije od vLLM-a. Sluzi kao provera, ne kao referenca brzine GPU-a.")
json.dump(res, open(os.path.join(a.out, f"{tag}_e2e.json"), "w"), indent=2)
print(json.dumps(res, indent=2))
