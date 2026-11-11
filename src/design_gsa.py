"""Robustness of the four ANN-designed protocols (Table 7) to parameter uncertainty.
Each design (wavelength, beam, cooling, irradiance, interval) is kept fixed; the radiobiological inputs of gsa.py and
the uncertain tissue inputs (perfusion and GNR absorption x0.5-2 / x0.67-1.33 around the design values, tissue
scattering and absorption x0.8-1.2) are sampled with a scrambled Sobol sequence (delta-P1 + Pennes)."""
import os, sys, json, time; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy.stats import qmc
from multiprocessing import Pool
import design_sim as ds, coupled as C, gsa

D0 = dict(depth=2.5, Dt=9.0, gnr=0.15, perf=1.5e-3)
DES = json.load(open("../data/design_trt.json"))
BIO = ["ra", "rb", "mu", "ab", "a37", "direct", "ab_n", "mu_n", "rho"]
PHYS = [("kperf", 0.5, 2.0, "log"), ("kgnr", 0.667, 1.333, "lin"), ("fs", 0.8, 1.2, "lin"), ("fa", 0.8, 1.2, "lin")]
SPACE = [s for s in gsa.SPACE if s[0] in BIO] + PHYS

def scale(u):
    o = {}
    for (n, lo, hi, s), x in zip(SPACE, u):
        o[n] = lo + x * (hi - lo) if s == "lin" else float(np.exp(np.log(lo) + x * (np.log(hi) - np.log(lo))))
    return o

def run(arg):
    key, x = arg; d = DES[key]
    lam = int(d["lam_nm"]); I = d["f_mpe"] * C.MPE[lam]
    for l in ds.MUA: ds.MUA[l] = {k: v * x["fa"] for k, v in gsa.MUA0[l].items()}
    for k, (a, b) in gsa.SCAT0.items(): ds.SCAT[k] = (a * x["fs"], b)
    Th, tum, skin = C.session(lam, I, d["spot_mm"], D0["depth"], D0["Dt"], D0["gnr"] * x["kgnr"], D0["perf"] * x["kperf"], d["h_cool"])
    fat = ~tum & ~skin
    pt = dict(C.TUM, a37=x["a37"], ab37=x["ab"], ra=x["ra"], rb=x["rb"], mu=x["mu"])
    pn = dict(C.NRM, a37=x["a37"], ab37=x["ab_n"], ra=x["ra"], rb=x["rb"], mu=x["mu_n"])
    et, Tm = C.fraction_eqd(Th, pt, tint=d["tint_h"], direct=x["direct"] >= 0.5)
    en, _ = C.fraction_eqd(Th, pn, tint=d["tint_h"], direct=False)
    N, H, Dd = gsa.N_FX, gsa.N_HT, gsa.D; Vt = C.VC[tum]
    ter = C.pct((N - H) * Dd + H * et[tum], Vt, 0.02) / (N * Dd)
    ter_n = max(C.pct((N - H) * Dd + H * en[skin], C.VC[skin], 0.98), C.pct((N - H) * Dd + H * en[fat], C.VC[fat], 0.98)) / (N * Dd)
    gain = H * (et[tum] - Dd); V = Vt * 1e-3; b37 = x["a37"] / x["ab"]; z = np.zeros_like(gain)
    d50 = gsa.dq(z, V, x["a37"], b37, x["rho"], 0.5) - gsa.dq(gain, V, x["a37"], b37, x["rho"], 0.5)
    return dict(x, design=key, TER=ter, TER_n=ter_n, TGF=ter / ter_n, dD50=d50, Tskin=float(Th[:, skin].max()), Ttum_max=float(Th[:, tum].max()))

if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
    U = qmc.Sobol(len(SPACE), scramble=True, seed=11).random(N); xs = [scale(u) for u in U]
    nom = dict(ra=1.73, rb=0.41, mu=0.5, ab=17.9, a37=0.386, direct=1.0, ab_n=3.0, mu_n=1.0, rho=5e8, kperf=1.0, kgnr=1.0, fs=1.0, fa=1.0)
    out = {}
    for key in DES:
        r0 = run((key, nom)); t0 = time.time()
        with Pool(2) as P: df = pd.DataFrame(P.map(run, [(key, x) for x in xs], chunksize=16))
        bio = df[[s[0] for s in SPACE if s[0] in BIO]]
        o = dict(nominal={k: r0[k] for k in ("TER", "TER_n", "TGF", "dD50", "Tskin", "Ttum_max")})
        for y in ("TER", "TER_n", "TGF", "dD50", "Tskin", "Ttum_max"):
            o[y] = dict(p2_5=float(np.percentile(df[y], 2.5)), median=float(df[y].median()), p97_5=float(np.percentile(df[y], 97.5)))
        o["P_TERn_le_1.02"] = float((df.TER_n <= 1.02).mean()); o["P_Tskin_le_43"] = float((df.Tskin <= 43).mean())
        o["P_Ttum_le_45"] = float((df.Ttum_max <= 45).mean()); o["P_TGF_gt_1"] = float((df.TGF > 1).mean())
        o["P_all_constraints"] = float(((df.TER_n <= 1.02) & (df.Tskin <= 43) & (df.Ttum_max <= 45)).mean())
        o["P_TER_gt_1.05"] = float((df.TER > 1.05).mean())
        o["P_TERn_le_1.02_if_rb_le_0.6"] = float((df.TER_n[df.rb <= 0.6] <= 1.02).mean())
        o["P_TERn_le_1.02_if_ra_le_1.9"] = float((df.TER_n[df.ra <= 1.9] <= 1.02).mean())
        o["P_TERn_le_1.03"] = float((df.TER_n <= 1.03).mean()); o["P_TERn_le_1.05"] = float((df.TER_n <= 1.05).mean())
        o["spearman_TERn"] = {n: float(df[n].corr(df.TER_n, method="spearman")) for n in [s[0] for s in SPACE]}
        out[key] = o; df.to_csv(f"../data/design_gsa_{key.replace('|', '_')}.csv.gz", index=False, compression="gzip")
        print(key, round(time.time() - t0), "s", json.dumps({k: v for k, v in o.items() if k.startswith("P_") or k == "nominal"}, default=lambda v: round(v, 3)), flush=True)
    json.dump(out, open("../data/design_gsa.json", "w"), indent=1, default=float)
