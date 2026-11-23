# CRYODYNE™ — arhitektura čipa pre RTL-a

Radni repozitorijum za korake 0–9 projektovanja arhitekture CRYODYNE AI čipa
(hibridni SSM/linearna rekurencija + atencija, ternarni i sub-1-bit formati).

## Struktura

| Direktorijum | Korak | Sadržaj | Status |
| --- | --- | --- | --- |
| `step0/` | 0 — skup opterećenja | `workloads.yaml` (modeli, formati, scenariji), `footprint.py` (memorijski otisak) | gotovo, v0.1 |
| `step1/` | 1 — karakterizacija na GPU-u | merenje slojeva na stvarnim dimenzijama, ceo Nemotron-H 8B, tragovi za Chakra | čeka pokretanje na GPU-u |
| `step2/` … `step9/` | 2–9 | dodaju se redom | — |

Prva kartica je za radne stanice (scenariji P1–P6 imaju prednost); serverska kartica dolazi druga.

## Pravila

- Svaki korak ima svoj direktorijum sa kodom, ulazima, rezultatima i `README.md` izveštajem.
- Promene pretpostavki idu u `step0/workloads.yaml`, nikad direktno u skripte kasnijih koraka.
- Svaki izveštaj razdvaja: **provereno** (izvor), **izračunato** (skripta), **procena**.
- Rezultati merenja sa iznajmljenih mašina čuvaju se u `stepN/results/<datum>_<mašina>/`.

Prateći dokument koraka 0: https://claude.ai/artifact/GwAijmibn4i3gXSZUeqNpf
