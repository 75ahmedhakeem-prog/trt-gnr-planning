"""Global uncertainty and sensitivity analysis of the reference case (808 nm, MPE, 25 mm spot, 9 mm tumour 2.5 mm deep).
Radiobiological (extended LQ) and physical (perfusion, GNR absorption, surface cooling, tissue optics) inputs are sampled
jointly with a scrambled Sobol sequence; first-order sensitivity indices are estimated from the same sample
(correlation ratio over quantile bins). Outputs: tumour TER (D98), TGF, dose gain 46*(TER-1), and the shift of the
total dose for 50% / 90% TCP (Poisson model with 10% inter-patient spread of alpha)."""
import os, sys, json, time; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.stats import qmc
from scipy.optimize import brentq
from multiprocessing import Pool
import design_sim as ds, coupled as C

MUA0 = {k: dict(v) for k, v in ds.MUA.items()}; SCAT0 = dict(ds.SCAT)
# name, low, high, scale ('lin' or 'log'); reference value in REF
SPACE = [
    ("ra",     1.20, 2.60, "lin"),   # alpha41/alpha37: SiHa 1.73, HeLa 2.55 [40]
    ("rb",     0.15, 1.00, "lin"),   # beta41/beta37: SiHa 0.42, HeLa 0.58, sigma points 0.18-0.66 [40]; 1 = no beta change
    ("mu",     0.25, 2.00, "log"),   # decay of sensitisation (1/h): clinical 0.5 [44,46]
    ("ab",     4.00, 20.0, "lin"),   # tumour alpha/beta (Gy): HeLa 5.0, SiHa 17.9 [40]
    ("a37",    0.20, 0.50, "lin"),   # tumour alpha37 (1/Gy): HeLa 0.23, SiHa 0.386 [40]
    ("direct", 0.00, 1.00, "lin"),   # direct heat kill switched off (<0.5) or on (>=0.5)
    ("ab_n",   2.00, 4.00, "lin"),   # normal-tissue alpha/beta (Gy): 3 [44]
    ("mu_n",   0.50, 2.00, "log"),   # normal-tissue decay (1/h): 1 [44]
    ("perf",   0.5e-3, 2.0e-3, "log"),  # tumour/tissue perfusion (1/s): 1e-3
    ("gnr",    0.10, 0.20, "lin"),   # GNR absorption (1/mm): 0.15
    ("h",      100., 1000., "log"),  # surface heat-transfer coefficient (W/m2/K): 200
    ("fs",     0.80, 1.20, "lin"),   # scaling of reduced scattering of all tissues
    ("fa",     0.80, 1.20, "lin"),   # scaling of background absorption of all tissues
    ("rho",    1e7, 1e9, "log"),     # clonogen density (1/cm3): 5e8
]
REF = dict(ra=1.73, rb=0.41, mu=0.5, ab=17.9, a37=0.386, direct=1.0, ab_n=3.0, mu_n=1.0, perf=1e-3, gnr=0.15, h=200.0,
           fs=1.0, fa=1.0, rho=5e8)
N_FX, N_HT, D = 23, 5, 2.0
GH_X, GH_W = np.polynomial.hermite_e.hermegauss(24); GH_W = GH_W / GH_W.sum()

def scale(u):
    out = {}
    for (n, lo, hi, s), x in zip(SPACE, u):
        out[n] = lo + x * (hi - lo) if s == "lin" else float(np.exp(np.log(lo) + x * (np.log(hi) - np.log(lo))))
    return out

def tcp(Dtot, gain_vox, V_cm3, a37, b37, rho, sig=0.1):
    eqd = Dtot + gain_vox                                            # EQD2 of each voxel
    f = 1 + sig * GH_X
    logS = -(a37 * f[:, None] + b37 * 2.0) * eqd[None, :]
    return float((GH_W * np.exp(-(rho * V_cm3[None, :] * np.exp(logS)).sum(1))).sum())

def dq(gain_vox, V, a37, b37, rho, q):
    return brentq(lambda d: tcp(d, gain_vox, V, a37, b37, rho) - q, 0.0, 300.0)

def run(x):
    for lam in ds.MUA: ds.MUA[lam] = {k: v * x["fa"] for k, v in MUA0[lam].items()}
    for k, (a, b) in SCAT0.items(): ds.SCAT[k] = (a * x["fs"], b)
    Th, tum, skin = C.session(808, C.MPE[808], 25, 2.5, 9.0, x["gnr"], x["perf"], x["h"])
    fat = ~tum & ~skin
    pt = dict(C.TUM, a37=x["a37"], ab37=x["ab"], ra=x["ra"], rb=x["rb"], mu=x["mu"])
    pn = dict(C.NRM, a37=x["a37"], ab37=x["ab_n"], ra=x["ra"], rb=x["rb"], mu=x["mu_n"])
    et, Tm = C.fraction_eqd(Th, pt, tint=0.5, direct=x["direct"] >= 0.5)
    en, _ = C.fraction_eqd(Th, pn, tint=0.5, direct=False)
    Vt = C.VC[tum]; tot = (N_FX - N_HT) * D + N_HT * et[tum]
    ter = C.pct(tot, Vt, 0.02) / (N_FX * D)
    ter_s = C.pct((N_FX - N_HT) * D + N_HT * en[skin], C.VC[skin], 0.98) / (N_FX * D)
    ter_f = C.pct((N_FX - N_HT) * D + N_HT * en[fat], C.VC[fat], 0.98) / (N_FX * D)
    gain = N_HT * (et[tum] - D); V = Vt * 1e-3; b37 = x["a37"] / x["ab"]; z = np.zeros_like(gain)
    d50 = dq(z, V, x["a37"], b37, x["rho"], 0.5) - dq(gain, V, x["a37"], b37, x["rho"], 0.5)
    d90 = dq(z, V, x["a37"], b37, x["rho"], 0.9) - dq(gain, V, x["a37"], b37, x["rho"], 0.9)
    return dict(x, T90=C.pct(Tm[tum], Vt, 0.10), T50=C.pct(Tm[tum], Vt, 0.5), Tskin=float(Th[:, skin].max()),
                Ttum_max=float(Th[:, tum].max()), TER=ter, TER_n=max(ter_s, ter_f), TGF=ter / max(ter_s, ter_f),
                gain_D98=46 * (ter - 1), dD50=d50, dD90=d90)

def first_order(X, y, nb=16):
    """correlation-ratio estimate of the first-order Sobol index"""
    o = {}; vy = y.var()
    for n in X.columns:
        x = X[n].values
        if n == "direct": b = (x >= 0.5).astype(int)
        else: b = np.minimum((np.argsort(np.argsort(x)) * nb) // len(x), nb - 1)
        m = np.array([y[b == i].mean() for i in np.unique(b)]); w = np.array([(b == i).mean() for i in np.unique(b)])
        s = float((w * (m - y.mean())**2).sum() / vy)
        o[n] = max(s - (len(np.unique(b)) - 1) / (len(x) - 1), 0.0)       # bias correction for finite bins
    return o

if __name__ == "__main__":
    import pandas as pd
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
    ref = run(REF); print("reference", {k: round(v, 3) for k, v in ref.items() if k in ("T50", "TER", "TGF", "gain_D98", "dD50", "dD90")}, flush=True)
    U = qmc.Sobol(len(SPACE), scramble=True, seed=7).random(N); xs = [scale(u) for u in U]
    t0 = time.time()
    with Pool(2) as P: res = P.map(run, xs, chunksize=16)
    print("runs", N, "time", round(time.time() - t0), "s", flush=True)
    df = pd.DataFrame(res); df.to_csv("../data/gsa_samples.csv.gz", index=False, compression="gzip")
    X = df[[s[0] for s in SPACE]]; out = {"reference": ref, "N": N, "space": SPACE, "summary": {}, "S1": {}, "spearman": {}}
    for y in ("TER", "TGF", "gain_D98", "dD50", "dD90", "TER_n", "T50", "Tskin"):
        v = df[y].values
        out["summary"][y] = dict(mean=v.mean(), median=float(np.median(v)), p2_5=float(np.percentile(v, 2.5)), p5=float(np.percentile(v, 5)),
                                 p95=float(np.percentile(v, 95)), p97_5=float(np.percentile(v, 97.5)), min=v.min(), max=v.max())
        out["S1"][y] = first_order(X, v); out["spearman"][y] = {n: float(pd.Series(X[n]).corr(df[y], method="spearman")) for n in X.columns}
    out["P_TER_gt_1.05"] = float((df.TER > 1.05).mean()); out["P_dD50_gt_5"] = float((df.dD50 > 5).mean())
    out["P_TERn_lt_1.02"] = float((df.TER_n < 1.02).mean()); out["P_TGF_gt_1"] = float((df.TGF > 1.0).mean())
    # biology only (physics at reference) and physics only (biology at reference), 1024 each
    for tag, keys in (("bio_only", ["ra", "rb", "mu", "ab", "a37", "direct", "ab_n", "mu_n", "rho"]), ("phys_only", ["perf", "gnr", "h", "fs", "fa"])):
        sub = [dict(REF, **{k: x[k] for k in keys}) for x in xs[:1024]]
        with Pool(2) as P: r2 = pd.DataFrame(P.map(run, sub, chunksize=16))
        out[tag] = {y: dict(p2_5=float(np.percentile(r2[y], 2.5)), median=float(r2[y].median()), p97_5=float(np.percentile(r2[y], 97.5)))
                    for y in ("TER", "TGF", "dD50", "T50")}
    json.dump(out, open("../data/gsa.json", "w"), indent=1, default=float)
    print(json.dumps({k: out[k] for k in ("summary",)}, indent=1, default=lambda v: round(float(v), 3)))
