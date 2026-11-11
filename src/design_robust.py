"""Robust irradiance for the maximum-TER designs: temperatures scale linearly with irradiance for fixed tissue
properties, so each uncertainty sample is simulated once at unit irradiance and evaluated with the irradiance scaled to 40-100% of the design value.
Reports the fraction of samples meeting all planning constraints (T_skin,max <= 43 C, TER_n <= 1.02,
T_tum,max <= 45 C) and the median tumour TER and dose gain at each scaling."""
import os, sys, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.stats import qmc
from multiprocessing import Pool
import design_sim as ds, coupled as C, gsa, design_gsa as G
F = np.round(np.arange(0.40, 1.0001, 0.05), 3)

def run(arg):
    key, x = arg; d = G.DES[key]; lam = int(d["lam_nm"]); Imax = d["f_mpe"] * C.MPE[lam]   # design irradiance
    for l in ds.MUA: ds.MUA[l] = {k: v * x["fa"] for k, v in gsa.MUA0[l].items()}
    for k, (a, b) in gsa.SCAT0.items(): ds.SCAT[k] = (a * x["fs"], b)
    Th1, tum, skin = C.session(lam, 1.0, d["spot_mm"], G.D0["depth"], G.D0["Dt"], G.D0["gnr"] * x["kgnr"], G.D0["perf"] * x["kperf"], d["h_cool"])
    fat = ~tum & ~skin; Vt = C.VC[tum]; N, H, Dd = gsa.N_FX, gsa.N_HT, gsa.D
    pt = dict(C.TUM, a37=x["a37"], ab37=x["ab"], ra=x["ra"], rb=x["rb"], mu=x["mu"])
    pn = dict(C.NRM, a37=x["a37"], ab37=x["ab_n"], ra=x["ra"], rb=x["rb"], mu=x["mu_n"])
    out = []
    for f in F:
        Th = 37 + f * Imax * (Th1 - 37)
        et, _ = C.fraction_eqd(Th, pt, tint=d["tint_h"], direct=x["direct"] >= 0.5); en, _ = C.fraction_eqd(Th, pn, tint=d["tint_h"], direct=False)
        ter = C.pct((N - H) * Dd + H * et[tum], Vt, 0.02) / (N * Dd)
        tn = max(C.pct((N - H) * Dd + H * en[skin], C.VC[skin], 0.98), C.pct((N - H) * Dd + H * en[fat], C.VC[fat], 0.98)) / (N * Dd)
        ok = (Th[:, skin].max() <= 43) and (tn <= 1.02) and (Th[:, tum].max() <= 45)
        out.append((ter, tn, ok, 46 * (ter - 1)))
    return out

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    U = qmc.Sobol(len(G.SPACE), scramble=True, seed=11).random(n); xs = [G.scale(u) for u in U]
    res = {}
    for key in ("808|max_TER", "1064|max_TER"):
        with Pool(2) as P: R = np.array(P.map(run, [(key, x) for x in xs], chunksize=8))   # (n, nF, 4)
        res[key] = {str(f): dict(P_all=float(R[:, i, 2].mean()), TER_median=float(np.median(R[:, i, 0])), gain_D98_median=float(np.median(R[:, i, 3])),
                                 TERn_p95=float(np.percentile(R[:, i, 1], 95))) for i, f in enumerate(F)}
        for f in F: print(key, f, {k: round(v, 3) for k, v in res[key][str(f)].items()}, flush=True)
    json.dump(res, open("../data/design_robust.json", "w"), indent=1)
