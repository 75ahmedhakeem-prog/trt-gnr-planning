"""Fit and prediction times of the ANN and GBT on the random split of ml_trt.py (one core)."""
import json, time, numpy as np, pandas as pd, ml_trt as M
d = pd.read_csv("../data/trt_dataset_v2.csv.gz"); d = d[d.T_tum_max <= 45.0].reset_index(drop=True); X = M.prep(d).values
rng = np.random.default_rng(M.SEED); y = d.TER_D98.values; idx = rng.permutation(len(d)); n = int(0.8 * len(d)); tr, te = idx[:n][:60000], idx[n:][:20000]
out = {}
for nm, mk in (("ANN", M.ann), ("GBT", M.gbt)):
    t0 = time.perf_counter(); m = mk().fit(X[tr], y[tr]); t1 = time.perf_counter(); p = m.predict(X[te]); t2 = time.perf_counter()
    out[nm] = dict(M.metr(y[te], p), t_fit=t1 - t0, t_pred_us=(t2 - t1) / len(te) * 1e6); print(nm, out[nm], flush=True)
json.dump(out, open("../data/time_ann.json", "w"), indent=1)
