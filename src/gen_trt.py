"""Thermoradiotherapy design dataset: delta-P1 + Pennes (60-min GNR heating) coupled to the extended LQ model."""
import os
import itertools, time, numpy as np, pandas as pd, sys
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import design_sim as ds, coupled as C
from trbio import lq, k_heat
LAM = [808, 1064]; SPOT = [15, 20, 25]; HC = [200.0, 1000.0]; DEPTH = [0, 1, 2, 3, 4, 5]; DT = [5.0, 7.5, 10.0, 12.5, 15.0]
PERF = [5e-4, 1e-3, 2e-3, 3e-3]; GNR = [0.05, 0.1, 0.2, 0.5]
FMPE = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]; TINT = [0.0, 0.5, 1.0, 2.0, 4.0]; N_FX, N_HT, d = 23, 5, 2.0
def stats(Th, tum, skin, tint):
    fat = ~tum & ~skin; Tm = Th.mean(0); KD = k_heat(Th, C.TUM).sum(0) * 60.0
    pt, pn = C.TUM, C.NRM
    a, b = lq(Tm, tint, pt); den = pt["a37"] + pt["a37"] / pt["ab37"] * 2
    et = (a * d + b * d * d) / den + KD / den
    a, b = lq(Tm, tint, pn); denn = pn["a37"] + pn["a37"] / pn["ab37"] * 2; en = (a * d + b * d * d) / denn
    Vt = C.VC[tum]; tot = (N_FX - N_HT) * d + N_HT * et
    nrm = (N_FX - N_HT) * d + N_HT * en; mask = skin | fat
    return dict(TER_D98=C.pct(tot[tum], Vt, 0.02) / (N_FX * d), TER_mean=float((tot[tum] * Vt).sum() / Vt.sum()) / (N_FX * d),
                TER_normal_D2=C.pct(nrm[mask], C.VC[mask], 0.98) / (N_FX * d),
                T90=C.pct(Tm[tum], Vt, 0.10), T50=C.pct(Tm[tum], Vt, 0.5))
def one(args):
    lam, S, h, dep, Dt, perf, g = args
    k, rhoc, wrc, tum, skin = ds.thermal_maps(dep, Dt, perf); Q = ds.light(lam, S, dep, Dt, g)
    F = ds.heat_fields(Q, k, rhoc, wrc, t_end=3600.0, every=60.0, h_top=h)[1:]          # 60 frames, 1-min spacing
    rows = []
    for f in FMPE:
        I = f * C.MPE[lam]; Th = 37.0 + I * F
        base = dict(lam_nm=lam, spot_mm=S, h_cool=h, depth_mm=dep, Dt_mm=Dt, perf=perf, mua_gnr=g, f_mpe=f, I_Wcm2=I,
                    T_skin_max=float(Th[:, skin].max()), T_normal_max=float(Th[:, ~tum].max()), T_tum_max=float(Th[:, tum].max()))
        for ti in TINT: rows.append({**base, "tint_h": ti, **stats(Th, tum, skin, ti)})
    return rows
if __name__ == "__main__":
    combos = list(itertools.product(LAM, SPOT, HC, DEPTH, DT, PERF, GNR)); t0 = time.time(); out = []
    with Pool(2) as p:
        for i, r in enumerate(p.imap_unordered(one, combos, chunksize=8)):
            out += r
            if i % 250 == 0: print(i, len(combos), round(time.time() - t0), flush=True)
    pd.DataFrame(out).to_csv("../data/trt_dataset_v2.csv.gz", index=False); print("done", len(out), round(time.time() - t0), flush=True)
