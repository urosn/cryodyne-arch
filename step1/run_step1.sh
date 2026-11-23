#!/usr/bin/env bash
# CRYODYNE korak 1 - jedna skripta koja sve pokrece na iznajmljenom GPU serveru.
# Upotreba:   bash run_step1.sh            (sve)
#             bash run_step1.sh --no-e2e   (bez preuzimanja modela od 16 GB)
# Rezultat:   cryodyne_step1_results_<datum>.tar.gz  -> posalji tu arhivu nazad.
set -euo pipefail
cd "$(dirname "$0")"
LOG=step1_log.txt
exec > >(tee -a "$LOG") 2>&1
echo "=== CRYODYNE korak 1, pocetak: $(date -u) ==="

nvidia-smi || { echo "GRESKA: nema NVIDIA GPU-a ili drajvera."; exit 1; }

# 1. Python okruzenje (odvojeno od sistema)
python3 -m venv .venv
source .venv/bin/activate
pip install -q --upgrade pip wheel
pip install -q torch pyyaml "transformers>=4.56" accelerate einops
# Optimizovani kerneli. Ako se neki ne instalira, merenje ide dalje i belezi 'nedostupno'.
pip install -q causal-conv1d mamba-ssm --no-build-isolation || echo "UPOZORENJE: mamba-ssm nije instaliran"
pip install -q flash-linear-attention || echo "UPOZORENJE: flash-linear-attention nije instaliran"
python3 -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,torch.cuda.get_device_name(0))"

# 2. Merenje slojeva na stvarnim dimenzijama (ne preuzima tezine)
python3 bench_layers.py --workloads workloads.yaml --out results

# 3. Ceo mali model + trag izvrsavanja za Chakra (preuzima ~16 GB sa Hugging Face-a)
if [[ "${1:-}" != "--no-e2e" ]]; then
  python3 e2e_small.py --model nvidia/Nemotron-H-8B-Base-8K --out results \
    || echo "UPOZORENJE: e2e merenje nije uspelo - pogledaj log"
fi

# 4. Pakovanje
nvidia-smi -q > results/nvidia_smi.txt
pip freeze > results/pip_freeze.txt
OUT="cryodyne_step1_results_$(date +%Y%m%d_%H%M).tar.gz"
tar czf "$OUT" results "$LOG"
echo "=== Gotovo: $OUT ($(du -h "$OUT" | cut -f1)). Preuzmi je i posalji nazad. ==="
