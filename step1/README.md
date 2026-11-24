# CRYODYNE™ — korak 1: merenje na GPU-u

Ovaj paket meri kako se slojevi modela iz skupa opterećenja (korak 0) ponašaju na stvarnom
GPU-u. Rezultati služe kao osnova za poređenje u koracima 2, 5 i 6 i kao ulaz za korak 7.

## Šta paket radi

1. **`bench_layers.py`** — za svaki tip sloja (Mamba-2, Gated DeltaNet, KDA, atencija, MLA,
   MoE, MLP) svakog modela meri projekcije (GEMM) i jezgro sloja, u prefill-u (4096 tokena)
   i u koraku dekodiranja za pet kombinacija batch/kontekst. Težine su slučajne, pa **ništa
   se ne preuzima** — slojevi se prave na stvarnim dimenzijama iz `workloads.yaml`.
2. **`e2e_small.py`** — meri ceo Nemotron-H 8B (preuzima ~16 GB) i snima trag izvršavanja
   (PyTorch ET + Kineto), koji se u koraku 7 pretvara u Chakra format. Služi i kao provera:
   zbir slojeva iz tačke 1 treba da se slaže sa celim modelom.
3. **`layers_ref.py`** — referentne implementacije slojeva u čistom PyTorch-u. Koriste se kao
   rezervni put i kao zlatni model ponašanja za kasnije korake.
4. **`run_step1.sh`** — jedna skripta koja instalira sve, pokreće merenja i pakuje rezultate.

## Mašina

| Stavka | Preporuka | Napomena |
| --- | --- | --- |
| GPU | 1× NVIDIA H100 80 GB (SXM ili PCIe) | referenca za data centar; H200 ili A100 80 GB takođe rade |
| CPU / RAM | ≥ 16 jezgara, ≥ 64 GB | prevođenje mamba-ssm je zahtevno za CPU |
| Disk | ≥ 100 GB | model 16 GB + keš paketa |
| OS | Ubuntu 22.04 ili 24.04, CUDA drajver 12.4+ | većina provajdera nudi gotovu sliku „PyTorch + CUDA" |
| Trajanje | procena: H100 30–90 min, RTX 4090/5090 25–60 min | najveći deo je instalacija; prevođenje mamba-ssm je ograničeno na 30 min |

Gde se iznajmljuje (na sat, bez dugoročnog ugovora): Lambda, RunPod, Vast.ai, a u EU na primer
Scaleway ili Nebius. Cene se menjaju svakodnevno, pa ih proveri na dan iznajmljivanja.

**Opciono, za scenarije individualnih korisnika:** isti paket pusti i na jednoj RTX 4090 ili
5090 (24/32 GB). Tada se vidi kako se slojevi ponašaju na potrošačkoj kartici, protiv koje će
se meriti CRYODYNE kartica. Pokreni sa `--no-e2e` ako kartica ima 24 GB.

## Uputstvo korak po korak

1. Iznajmi mašinu iz tabele i poveži se na nju preko SSH-a (provajder daje komandu).
2. Prebaci paket na mašinu, sa svog računara:
   ```
   scp cryodyne_step1.tar.gz korisnik@ADRESA:~
   ```
3. Na mašini:
   ```
   tar xzf cryodyne_step1.tar.gz && cd step1
   bash run_step1.sh
   ```
   Ako se Nemotron-H ne preuzme zbog licence: na huggingface.co prihvati licencu modela,
   napravi token (Settings → Access Tokens) i pre pokretanja izvrši `export HF_TOKEN=...`.
   Ako ne želiš preuzimanje modela, pokreni `bash run_step1.sh --no-e2e`.
4. Kada skripta ispiše „Gotovo", preuzmi arhivu na svoj računar:
   ```
   scp korisnik@ADRESA:~/step1/cryodyne_step1_results_*.tar.gz .
   ```
5. **Ugasi mašinu** kod provajdera (inače se naplaćuje dalje).
6. Pošalji mi arhivu u razgovoru. Ja iz nje pravim izveštaj koraka 1 i ažuriram korak 0.

## Šta je provereno, a šta ne

- Kod je testiran na CPU-u sa smanjenim dimenzijama (`python3 bench_layers.py --smoke`):
  radi od početka do kraja i proizvodi `layers.csv`.
- **Na GPU-u nije testiran**, jer pripremno okruženje nema GPU. Najverovatnija mesta za
  grešku su pozivi optimizovanih kernela (`mamba_ssm`, `fla`), čiji se potpisi menjaju među
  verzijama. Ako neki poziv padne, skripta beleži grešku u `layers.csv` i nastavlja; tada
  mi pošalji `step1_log.txt` i ispraviću poziv.
- Vremena u koloni `impl = ref` su referentna implementacija i **ne** mere hardver.
- MoE se meri preko `bmm` nad aktivnim ekspertima, bez rutiranja i bez optimizovanog
  grupnog GEMM-a; to je donja procena troška rutiranja.
