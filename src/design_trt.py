"""ANN-based protocol design for an unseen tumour; verification with delta-P1 and Monte Carlo light transport."""
import os
import sys, json, itertools, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import design_sim as ds, mc_het, coupled as C, ml_trt as M, gen_trt as G
d = pd.read_csv("../data/trt_dataset_v2.csv.gz"); d = d[d.T_tum_max <= 45.0].reset_index(drop=True); X = M.prep(d)
TG = ["TER_D98", "TER_normal_D2", "T_skin_max", "T_tum_max"]
models = {tg: [M.ann(s).fit(X, d[tg].values) for s in (42, 7, 123)] for tg in TG}
REF = dict(depth_mm=2.5, Dt_mm=9.0, mua_gnr=0.15, perf=1.5e-3)
grid = dict(lam_nm=[808, 1064], spot_mm=np.arange(15, 25.01, 1.0), h_cool=[200.0, 1000.0], f_mpe=np.arange(0.1, 1.0001, 0.025), tint_h=[0.5, 1.0, 2.0])
G_ = pd.DataFrame(list(itertools.product(*grid.values())), columns=list(grid))
for k, v in REF.items(): G_[k] = v
XG = M.prep(G_)
for tg in TG:
    P = np.array([m.predict(XG) for m in models[tg]]); G_[tg] = P.mean(0); G_[tg + "_sd"] = P.std(0, ddof=1)
ok = (G_.T_skin_max <= 43.0) & (G_.TER_normal_D2 <= 1.02) & (G_.T_tum_max <= 45.0)
G_[ok].to_csv("../data/design_trt_feasible.csv", index=False)
res = {}
for lam in (808, 1064):
    s = G_[ok & (G_.lam_nm == lam)]
    best = s.loc[s.TER_D98.idxmax()].to_dict(); res[f"{lam}|max_TER"] = best
    s2 = s[s.TER_D98 >= 0.97 * best["TER_D98"]]; res[f"{lam}|min_power_97pct"] = s2.loc[(s2.f_mpe * (s2.spot_mm**2)).idxmin()].to_dict()
def exact(row, light="dP1"):
    lam, S, h, I = int(row["lam_nm"]), row["spot_mm"], row["h_cool"], row["f_mpe"] * C.MPE[int(row["lam_nm"])]
    k, rhoc, wrc, tum, skin = ds.thermal_maps(REF["depth_mm"], REF["Dt_mm"], REF["perf"])
    if light == "dP1": Q = ds.light(lam, S, REF["depth_mm"], REF["Dt_mm"], REF["mua_gnr"])
    else:
        mua, mus, g, _ = ds.optics(lam, REF["depth_mm"], REF["Dt_mm"], REF["mua_gnr"]); P = 0.01 * np.pi * (S / 2)**2
        m = mc_het.run(ds.R, ds.H, ds.NR, ds.NZ, mua, mus, g, S / 2, P=P, n_in=1.4, n_top=1.0, n_side=1.4, n_bot=1.4, N=100000, seed=11)
        vol = 2 * np.pi * ds.rc[:, None] * ds.DR * ds.DZ; E = (m["Q"] * vol).reshape(ds.TR, 2, ds.TZ, 2).sum((1, 3)); Q = E / (2 * np.pi * ds.rt[:, None] * 0.25) * 1e9
    F = ds.heat_fields(Q, k, rhoc, wrc, t_end=3600.0, every=60.0, h_top=h)[1:]; Th = 37 + I * F
    st = G.stats(Th, tum, skin, row["tint_h"])
    return dict(TER_D98=st["TER_D98"], TER_normal_D2=st["TER_normal_D2"], T_skin_max=float(Th[:, skin].max()), T_tum_max=float(Th[:, tum].max()), T90=st["T90"], T50=st["T50"])
for key, row in res.items():
    row["dP1"] = exact(row, "dP1"); row["MC"] = exact(row, "MC")
    print(key, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if not isinstance(v, dict)}, "\n  dP1", {k: round(v, 3) for k, v in row["dP1"].items()}, "\n  MC ", {k: round(v, 3) for k, v in row["MC"].items()}, flush=True)
json.dump(res, open("../data/design_trt.json", "w"), indent=1, default=float)
