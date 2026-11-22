"""Robust protocol design under uncertain perfusion and GNR loading: ANN surrogate vs exact delta-P1 + Pennes.
Uncertain inputs (as in design_gsa.py): perfusion x0.5-2 (log) and mu_a,GNR x0.67-1.33 around the unseen tumour,
NS scrambled-Sobol samples.  A candidate of the 4884-point grid of design_trt.py is feasible when all three planning
constraints hold in >= 90% of the samples; among feasible candidates the one with the highest median tumour TER is chosen.
Exact search: 44 coupled simulations per sample (44 x NS in total).  ANN search: 4884 x NS predictions per target."""
import os, sys, json, time, itertools; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy.stats import qmc
from multiprocessing import Pool
import design_sim as ds, coupled as C, gen_trt as G, ml_trt as M
D0 = dict(depth_mm=2.5, Dt_mm=9.0, mua_gnr=0.15, perf=1.5e-3)
LAM, SPOT, HC = [808, 1064], list(np.arange(15, 25.01, 1.0)), [200.0, 1000.0]
FM = np.round(np.arange(0.1, 1.0001, 0.025), 4); TI = [0.5, 1.0, 2.0]; Q90 = 0.90
TG = ["TER_D98", "TER_normal_D2", "T_skin_max", "T_tum_max"]

def samples(ns):
    U = qmc.Sobol(2, scramble=True, seed=11).random(ns)
    return np.exp(np.log(0.5) + U[:, 0] * np.log(4.0)), 0.667 + U[:, 1] * 0.666

def exact_one(arg):
    lam, S, h, kp, kg = arg
    k, rhoc, wrc, tum, skin = ds.thermal_maps(D0["depth_mm"], D0["Dt_mm"], D0["perf"] * kp); Q = ds.light(lam, S, D0["depth_mm"], D0["Dt_mm"], D0["mua_gnr"] * kg)
    F = ds.heat_fields(Q, k, rhoc, wrc, t_end=3600.0, every=60.0, h_top=h)[1:]; rows = []
    for f in FM:
        Th = 37 + f * C.MPE[lam] * F; Tsk = float(Th[:, skin].max()); Ttm = float(Th[:, tum].max())
        for ti in TI:
            st = G.stats(Th, tum, skin, ti); rows.append((lam, S, h, f, ti, st["TER_D98"], st["TER_normal_D2"], Tsk, Ttm))
    return rows

def choose(df, ns):
    """df: one row per (candidate, sample) with the four targets; returns the robust optimum per wavelength."""
    df = df.assign(ok=(df.T_skin_max <= 43) & (df.TER_normal_D2 <= 1.02) & (df.T_tum_max <= 45))
    g = df.groupby(["lam_nm", "spot_mm", "h_cool", "f_mpe", "tint_h"]).agg(p_ok=("ok", "mean"), TER_med=("TER_D98", "median"), TER_p05=("TER_D98", lambda x: np.percentile(x, 5))).reset_index()
    out = {}
    for lam in LAM:
        s = g[(g.lam_nm == lam) & (g.p_ok >= Q90)]; out[lam] = s.loc[s.TER_med.idxmax()].to_dict() if len(s) else None
    return out, g

if __name__ == "__main__":
    NS = int(sys.argv[1]) if len(sys.argv) > 1 else 128; kp, kg = samples(NS); res = dict(NS=NS)
    # ---- ANN ----
    d = pd.read_csv("../data/trt_dataset_v2.csv.gz"); d = d[d.T_tum_max <= 45.0].reset_index(drop=True); X = M.prep(d)
    t0 = time.time(); models = {tg: M.ann(42).fit(X, d[tg].values) for tg in TG}; res["t_train_ann_s"] = time.time() - t0
    cand = pd.DataFrame(list(itertools.product(LAM, SPOT, HC, FM, TI)), columns=["lam_nm", "spot_mm", "h_cool", "f_mpe", "tint_h"])
    big = pd.concat([cand.assign(perf=D0["perf"] * a, mua_gnr=D0["mua_gnr"] * b, depth_mm=D0["depth_mm"], Dt_mm=D0["Dt_mm"], sample=i) for i, (a, b) in enumerate(zip(kp, kg))], ignore_index=True)
    t0 = time.time(); XB = M.prep(big)
    for tg in TG: big[tg] = models[tg].predict(XB)
    res["t_ann_search_s"] = time.time() - t0; res["n_ann_pred"] = len(big) * len(TG)
    ann_best, g_ann = choose(big, NS); print("ANN robust", ann_best, round(res["t_ann_search_s"], 1), "s", flush=True)
    # ---- exact ----
    args = [(lam, S, h, a, b) for a, b in zip(kp, kg) for lam, S, h in itertools.product(LAM, SPOT, HC)]
    t0 = time.time()
    with Pool(int(os.environ.get("NPROC", 1))) as P: rows = [r for rr in P.imap(exact_one, args, chunksize=4) for r in rr]
    res["t_exact_search_s"] = time.time() - t0; res["n_exact_sim"] = len(args)
    ex = pd.DataFrame(rows, columns=["lam_nm", "spot_mm", "h_cool", "f_mpe", "tint_h"] + TG)
    ex_best, g_ex = choose(ex, NS); print("exact robust", ex_best, round(res["t_exact_search_s"]), "s", flush=True)
    # ---- exact evaluation of the ANN robust choice ----
    key = ["lam_nm", "spot_mm", "h_cool", "f_mpe", "tint_h"]; gx = g_ex.set_index(key)
    for lam in LAM:
        a = ann_best[lam]; e = ex_best[lam]; ae = gx.loc[tuple(a[k] for k in key)].to_dict() if a else None
        res[str(lam)] = dict(ann=a, exact=e, ann_choice_exact=ae, regret_TER_med=(e["TER_med"] - ae["TER_med"]) if a and e else None)
        print(lam, "ANN choice exact:", ae, "regret", res[str(lam)]["regret_TER_med"], flush=True)
    json.dump(res, open("../data/robust_design.json", "w"), indent=1, default=float)
