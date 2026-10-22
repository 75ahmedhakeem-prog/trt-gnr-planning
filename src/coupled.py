"""Coupled GNR photothermal heating (delta-P1 + Pennes) and thermoradiobiology (extended LQ)."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, design_sim as ds
from trbio import lq, k_heat
MPE = {808: 0.2 * 10**(0.002 * 108), 1064: 1.0}
# tumour: SiHa ML estimates (van Leeuwen 2017) with clinical decay mu = 0.5 1/h (van Leeuwen 2018)
TUM = dict(a37=0.386, ab37=17.9, ra=1.73, rb=0.41, mu=0.5, dS=392.08, dH=147907.8)
# normal tissue: same enhancement, faster decay (mu = 1 1/h), alpha/beta = 3 Gy, no direct kill (van Leeuwen 2018)
NRM = dict(a37=0.386, ab37=3.0, ra=1.73, rb=0.41, mu=1.0, dS=392.08, dH=147907.8)
VC = (2 * np.pi * ds.rt[:, None] * 0.5 * 0.5) * np.ones((1, ds.TZ))     # mm^3 per thermal cell
def session(lam, I, spot, depth, Dt, gnr, perf, h, t_ht=3600.0, every=30.0):
    """temperature history (n, TR, TZ) during a heating session of t_ht s at irradiance I (W/cm2)"""
    k, rhoc, wrc, tum, skin = ds.thermal_maps(depth, Dt, perf)
    Q = ds.light(lam, spot, depth, Dt, gnr)
    F = ds.heat_fields(Q, k, rhoc, wrc, t_end=t_ht, every=every, h_top=h)
    return 37.0 + I * F, tum, skin
def fraction_eqd(Th, p, d=2.0, tint=0.5, dref=2.0, every=30.0, direct=True):
    """EQD (Gy, re dref) of one d-Gy fraction combined with the heating session Th (time history)"""
    Tm = Th.mean(0)                                        # session-mean temperature drives radiosensitisation
    a, b = lq(Tm, tint, p); a37 = p["a37"]; b37 = a37 / p["ab37"]; den = a37 + b37 * dref
    e = (a * d + b * d**2) / den
    if direct: e = e + (k_heat(Th, p).sum(0) * every) / den        # exact Arrhenius integral of direct kill
    return e, Tm
def pct(x, w, q):
    o = np.argsort(x); c = np.cumsum(w[o]) / w.sum(); return float(x[o][min(np.searchsorted(c, q), len(x) - 1)])
def tcp(eqd_tot, V_mm3, p, rho=5e8, sig_rel=0.1, n_mc=400, seed=0, dref=2.0):
    """Poisson TCP with inter-patient spread of alpha (sigma = 10%); rho clonogens per cm^3; voxel EQD (Gy) and volume (mm^3)"""
    rng = np.random.default_rng(seed); a37 = p["a37"]; b37 = a37 / p["ab37"]
    V = V_mm3 * 1e-3; out = []
    for f in rng.normal(1, sig_rel, n_mc):
        logS = -(a37 * f + b37 * dref) * eqd_tot
        out.append(np.exp(-(rho * V * np.exp(logS)).sum()))
    return float(np.mean(out))
