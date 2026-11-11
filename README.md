# trt-gnr-planning

Code and data for the manuscript **"Gold nanorods as heaters, not dose enhancers: planning mild thermoradiotherapy within skin exposure limits"** (A.M. Abdel Hakeem, S.A. Ahmed).

The framework couples
1. Gans optics of gold nanorods (`gans.py`, `goldconc.py`),
2. δ-Eddington (δP1) light transport (`dp1.py`) with Monte Carlo reference codes (`mc.py`, `mc_het.py`),
3. the Pennes bioheat equation on an axisymmetric skin–fat–tumour model (`design_sim.py`),
4. the extended linear-quadratic thermoradiobiology model of van Leeuwen et al. (`trbio.py`, `coupled.py`),
5. a 288,000-sample design dataset (`gen_trt.py`) and ANN / gradient-boosted-tree surrogates (`ml_trt.py`, `design_trt.py`).

## Requirements
Python ≥ 3.10, `numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib`, `pillow` (see `requirements.txt`).

## Reproducing the results (run inside `src/`)
| Step | Script | Output |
|---|---|---|
| Verification | `verify_heat.py`, `verify_mc.py`, `trbio.py` | `data/verify_*.json`, benchmark printout |
| Reference case and TCP | `step1.py`, `step1_tcp.py` | `data/step1*.json` |
| Time interval | `step2.py` | `data/step2.json` |
| Gold dose enhancement | `def_gold.py`, `step3.py` | `data/step3*.json` |
| Dataset (≈ 20 min on 2 cores) | `gen_trt.py` | `data/trt_dataset_v2.csv` (provided gzipped) |
| Surrogates | `ml_trt.py` | `data/ml_trt_v2.json`, `data/ml_trt_holdout_v2.csv` |
| Treatability maps | `trt_maps.py` | `data/trt_maps.json` |
| Protocol design + δP1/MC verification | `design_trt.py` | `data/design_trt.json` |
| Global uncertainty analysis (4096 sets, ≈ 15 min on 2 cores) | `gsa.py` | `data/gsa.json`, `data/gsa_samples.csv.gz` |
| In vivo interval tests (Overgaard 1980; Folefac et al. 2025) | `bench_overgaard.py` | `data/bench_overgaard.json` |
| Figures | `fig_*.py`, `fig_gsa.py`, `make_ga_trt.py` | `figures/*.png` |

`data/nist/` holds the NIST mass attenuation and energy-absorption coefficients (Hubbell & Seltzer, NISTIR 5632) for Au, water, Al and Cu.

## Licence
MIT
