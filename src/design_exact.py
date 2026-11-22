"""Exhaustive exact (delta-P1 + Pennes) protocol search for the unseen tumour of design_trt.py, and cost accounting.
Temperature rise is linear in irradiance and the interval enters only the radiobiology, so the 4884-point design grid
of design_trt.py needs only one coupled simulation per (wavelength, beam diameter, cooling) = 44 simulations."""
import os, sys, json, time, itertools; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import design_sim as ds, coupled as C, gen_trt as G
REF = dict(depth_mm=2.5, Dt_mm=9.0, mua_gnr=0.15, perf=1.5e-3)
grid = dict(lam_nm=[808, 1064], spot_mm=np.arange(15, 25.01, 1.0), h_cool=[200.0, 1000.0])
FM = np.arange(0.1, 1.0001, 0.025); TI = [0.5, 1.0, 2.0]
if __name__ == "__main__":
    rows = []; t_sim = []; t_post = []
    for lam, S, h in itertools.product(*grid.values()):
        t0 = time.perf_counter()
        k, rhoc, wrc, tum, skin = ds.thermal_maps(REF["depth_mm"], REF["Dt_mm"], REF["perf"]); Q = ds.light(lam, S, REF["depth_mm"], REF["Dt_mm"], REF["mua_gnr"])
        F = ds.heat_fields(Q, k, rhoc, wrc, t_end=3600.0, every=60.0, h_top=h)[1:]; t1 = time.perf_counter(); t_sim.append(t1 - t0)
        for f, ti in itertools.product(FM, TI):
            Th = 37 + f * C.MPE[lam] * F; st = G.stats(Th, tum, skin, ti)
            rows.append(dict(lam_nm=lam, spot_mm=S, h_cool=h, f_mpe=round(f, 4), tint_h=ti, TER_D98=st["TER_D98"], TER_normal_D2=st["TER_normal_D2"],
                             T_skin_max=float(Th[:, skin].max()), T_tum_max=float(Th[:, tum].max()), T90=st["T90"], T50=st["T50"]))
        t_post.append((time.perf_counter() - t1) / (len(FM) * len(TI)))
    E = pd.DataFrame(rows); ok = (E.T_skin_max <= 43) & (E.TER_normal_D2 <= 1.02) & (E.T_tum_max <= 45)
    A = pd.read_csv("../data/design_trt_feasible.csv"); des = json.load(open("../data/design_trt.json")); res = dict(n_sim=len(t_sim), n_grid=len(E),
        t_sim_mean_s=float(np.mean(t_sim)), t_post_mean_s=float(np.mean(t_post)), t_total_s=float(np.sum(t_sim) + np.sum(t_post) * len(FM) * len(TI)))
    for lam in (808, 1064):
        s = E[ok & (E.lam_nm == lam)]; b = s.loc[s.TER_D98.idxmax()].to_dict(); a = des[f"{lam}|max_TER"]
        # exact evaluation of the ANN optimum on the same grid point
        m = E[(E.lam_nm == lam) & np.isclose(E.spot_mm, a["spot_mm"]) & np.isclose(E.h_cool, a["h_cool"]) & np.isclose(E.f_mpe, round(a["f_mpe"], 4)) & np.isclose(E.tint_h, a["tint_h"])].iloc[0].to_dict()
        # how close to the exact optimum is the ANN choice, and is the ANN feasible set the exact one?
        fe = set(map(tuple, E[ok & (E.lam_nm == lam)][["spot_mm", "h_cool", "f_mpe", "tint_h"]].round(4).values))
        fa = set(map(tuple, A[A.lam_nm == lam][["spot_mm", "h_cool", "f_mpe", "tint_h"]].round(4).values))
        res[str(lam)] = dict(exact_best=b, ann_best_design={k: a[k] for k in ("spot_mm", "h_cool", "f_mpe", "tint_h", "TER_D98")}, ann_best_exact=m,
                             regret_TER=b["TER_D98"] - m["TER_D98"], n_feasible_exact=len(fe), n_feasible_ann=len(fa), n_both=len(fe & fa))
        print(lam, "exact best", {k: round(v, 4) for k, v in b.items()}, "\n   ANN choice evaluated exactly", {k: round(v, 4) for k, v in m.items()},
              "\n   regret", round(res[str(lam)]["regret_TER"], 4), "feasible exact/ANN/both", len(fe), len(fa), len(fe & fa), flush=True)
    print({k: v for k, v in res.items() if k.startswith("t_") or k.startswith("n_")})
    E.to_csv("../data/design_exact_grid.csv.gz", index=False); json.dump(res, open("../data/design_exact.json", "w"), indent=1, default=float)
