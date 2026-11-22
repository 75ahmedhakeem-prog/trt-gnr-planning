"""Fit and prediction times of all surrogates on the random split of ml_trt.py (tumour TER, 60,000 training and
20,000 test samples), on one core of an otherwise idle machine; median of three repetitions."""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[v] = "1"
import json, time, platform, numpy as np, pandas as pd, ml_trt as M, ml_baselines as B
d = pd.read_csv("../data/trt_dataset_v2.csv.gz"); d = d[d.T_tum_max <= 45.0].reset_index(drop=True); X = M.prep(d).values
rng = np.random.default_rng(M.SEED); y = d.TER_D98.values; idx = rng.permutation(len(d)); n = int(0.8 * len(d)); tr, te = idx[:n][:60000], idx[n:][:20000]
MK = {"ANN": M.ann, "GBT": M.gbt, **B.MODELS}; out = {}
for nm, mk in MK.items():
    tf, tp = [], []
    for rep in range(3):
        t0 = time.perf_counter(); m = mk().fit(X[tr], y[tr]); t1 = time.perf_counter(); m.predict(X[te]); t2 = time.perf_counter()
        tf.append(t1 - t0); tp.append((t2 - t1) / len(te) * 1e6)
    out[nm] = dict(t_fit=float(np.median(tf)), t_pred_us=float(np.median(tp)), t_fit_all=tf, t_pred_all=tp); print(nm, round(out[nm]["t_fit"], 2), round(out[nm]["t_pred_us"], 2), flush=True)
out["_machine"] = dict(cpu="Intel Xeon @ 2.10GHz", cores_used=1, python=platform.python_version())
json.dump(out, open("../data/time_all.json", "w"), indent=1)
