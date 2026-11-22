"""Heterogeneous gold-nanorod distributions at equal total gold (reviewer comment).
The GNR absorption coefficient in the tumour is mu_a,GNR(r,z) = mu_a,GNR,mean * c(r,z), where the relative
concentration c has a volume-weighted tumour mean of 1 (same total gold as the uniform case).  Patterns:
  rim (peripheral accumulation), shell (necrotic core without gold), centre (intratumoural injection),
  light-side / far-side gradients, and log-normal random fields (sigma_ln 0.5 / 1.0, correlation length 0.5 / 2 mm;
  axisymmetric, i.e. ring-shaped in 3D).  Inter-session variation: a new random field in each of the five sessions,
  and gold wash-out between sessions.  Light: delta-P1; heat: Pennes; biology: extended LQ (coupled.py)."""
import os, sys, json, time; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.ndimage import gaussian_filter
import design_sim as ds, dp1, coupled as C, gsa

N_FX, N_HT, D = 23, 5, 2.0
REF = dict(lam=808, f=1.0, S=25, h=200.0, tint=0.5, depth=2.5, Dt=9.0, gnr=0.15, perf=1e-3, name="Reference case")
DES = json.load(open("../data/design_trt.json"))

def geometry(depth, Dt):
    tum, _ = ds.regions(ds.rc, ds.zc, depth, Dt); zc0 = ds.SKIN + depth + Dt / 2
    rho = np.sqrt(ds.rc[:, None]**2 + (ds.zc[None, :] - zc0)**2) / (Dt / 2); dz = (ds.zc[None, :] - zc0) / (Dt / 2) * np.ones_like(rho)
    V = 2 * np.pi * ds.rc[:, None] * ds.DR * ds.DZ * np.ones_like(rho); return tum, rho, dz, V

def normalise(c, tum, V):
    c = np.where(tum, np.maximum(c, 0), 0.0); return c / ((c * V)[tum].sum() / V[tum].sum())

def pattern(kind, depth, Dt, seed=0, sig=0.5, ell=1.0):
    tum, rho, dz, V = geometry(depth, Dt); R = Dt / 2
    if kind == "uniform": c = np.ones_like(rho)
    elif kind == "rim": c = np.exp(-(1 - rho) * R / 0.75)                        # e-fold length 0.75 mm from the surface
    elif kind == "shell": c = (rho >= 0.5).astype(float)                       # necrotic core (inner half radius) without gold
    elif kind == "centre": c = np.exp(-rho**2 / (2 * 0.35**2))                 # intratumoural injection
    elif kind == "light-side": c = 1 - 0.8 * dz                                # more gold on the illuminated side
    elif kind == "far-side": c = 1 + 0.8 * dz
    elif kind == "lognormal":
        rng = np.random.default_rng(seed); w = gaussian_filter(rng.standard_normal(rho.shape), ell / ds.DR, mode="nearest")
        w = (w - w[tum].mean()) / w[tum].std(); c = np.exp(sig * w)
    else: raise ValueError(kind)
    return normalise(c, tum, V)

def light(lam, S, depth, Dt, mua_gnr_map, g=0.9):
    mua, mus, gg, tum = ds.optics(lam, depth, Dt, 0.0, g); mua = mua + mua_gnr_map
    o = dp1.solve(ds.R, ds.H, ds.NR, ds.NZ, mua, mus, gg, S / 2, P=0.01 * np.pi * (S / 2)**2, n_in=1.4)
    vol = 2 * np.pi * ds.rc[:, None] * ds.DR * ds.DZ; E = (o["Q"] * vol).reshape(ds.TR, 2, ds.TZ, 2).sum((1, 3))
    return E / (2 * np.pi * ds.rt[:, None] * 0.25) * 1e9

def session(case, c):
    lam = case["lam"]; I = case["f"] * C.MPE[lam]
    k, rhoc, wrc, tum, skin = ds.thermal_maps(case["depth"], case["Dt"], case["perf"])
    Q = light(lam, case["S"], case["depth"], case["Dt"], case["gnr"] * c)
    F = ds.heat_fields(Q, k, rhoc, wrc, t_end=3600.0, every=30.0, h_top=case["h"])
    return 37.0 + I * F, tum, skin

def evaluate(case, sessions):
    """sessions: list of five temperature histories (one per heated fraction)."""
    _, tum, skin = sessions[0]; fat = ~tum & ~skin; tot_t = (N_FX - N_HT) * D; tot_n = (N_FX - N_HT) * D; Tm_all = []
    for Th, _, _ in sessions:
        et, Tm = C.fraction_eqd(Th, C.TUM, tint=case["tint"]); en, _ = C.fraction_eqd(Th, C.NRM, tint=case["tint"], direct=False)
        tot_t = tot_t + et; tot_n = tot_n + en; Tm_all.append(Tm)
    Tm = np.mean(Tm_all, 0); Vt = C.VC[tum]
    ter = C.pct(tot_t[tum], Vt, 0.02) / (N_FX * D)
    tern = max(C.pct(tot_n[skin], C.VC[skin], 0.98), C.pct(tot_n[fat], C.VC[fat], 0.98)) / (N_FX * D)
    gain = tot_t[tum] - N_FX * D; a37 = C.TUM["a37"]; b37 = a37 / C.TUM["ab37"]; z = np.zeros_like(gain)
    d50 = gsa.dq(z, Vt * 1e-3, a37, b37, 5e8, 0.5) - gsa.dq(gain, Vt * 1e-3, a37, b37, 5e8, 0.5)
    Tmax = max(float(Th[:, tum].max()) for Th, _, _ in sessions); Tsk = max(float(Th[:, skin].max()) for Th, _, _ in sessions)
    return dict(TER_D98=ter, TER_mean=float((tot_t[tum] * Vt).sum() / Vt.sum()) / (N_FX * D), TER_n=tern, TGF=ter / tern, dD50=d50,
                T90=C.pct(Tm[tum], Vt, 0.10), T50=C.pct(Tm[tum], Vt, 0.5), Ttum_max=Tmax, Tskin_max=Tsk)

def capped(case, c_list):
    """evaluate as planned, then with the irradiance lowered (if needed) so that Ttum,max <= 45 C and Tskin,max <= 43 C."""
    memo = {}
    for c in c_list:
        if id(c) not in memo: memo[id(c)] = session(case, c)
    S = [memo[id(c)] for c in c_list]; r = evaluate(case, S)
    fac = min(1.0, (45 - 37) / (r["Ttum_max"] - 37), (43 - 37) / (r["Tskin_max"] - 37))
    S2 = [(37 + fac * (Th - 37), t, s) for Th, t, s in S]; r2 = evaluate(case, S2)
    return dict(planned=r, capped={**r2, "irr_factor": fac})

def cases():
    out = [REF]
    for key, nm in (("808|max_TER", "808 nm max. TER"), ("808|min_power_97pct", "808 nm low power"), ("1064|max_TER", "1064 nm max. TER"), ("1064|min_power_97pct", "1064 nm low power")):
        d = DES[key]; out.append(dict(lam=int(d["lam_nm"]), f=d["f_mpe"], S=d["spot_mm"], h=d["h_cool"], tint=d["tint_h"], depth=2.5, Dt=9.0, gnr=0.15, perf=1.5e-3, name=nm))
    return out

if __name__ == "__main__":
    NREAL = int(sys.argv[1]) if len(sys.argv) > 1 else 20; t0 = time.time(); res = {}
    for case in cases():
        o = {}; dep, Dt = case["depth"], case["Dt"]
        for kind in ("uniform", "rim", "shell", "centre", "light-side", "far-side"):
            o[kind] = capped(case, [pattern(kind, dep, Dt)] * N_HT)
        for sig in (0.5, 1.0):
            for ell in (0.5, 2.0):
                rs = [capped(case, [pattern("lognormal", dep, Dt, seed=s, sig=sig, ell=ell)] * N_HT) for s in range(NREAL)]
                o[f"lognormal s{sig} l{ell} fixed"] = rs
                rv = [capped(case, [pattern("lognormal", dep, Dt, seed=1000 + 10 * s + j, sig=sig, ell=ell) for j in range(N_HT)]) for s in range(NREAL)]
                o[f"lognormal s{sig} l{ell} varying"] = rv
        wash = [1.0, 0.85, 0.72, 0.61, 0.52]                                           # ~15% gold loss per session
        o["washout 15%/session"] = capped(case, [pattern("uniform", dep, Dt) * w for w in wash])
        res[case["name"]] = o
        u = o["uniform"]["planned"]
        print(case["name"], round(time.time() - t0), "s  uniform TER", round(u["TER_D98"], 4), "rim", round(o["rim"]["planned"]["TER_D98"], 4),
              "shell", round(o["shell"]["planned"]["TER_D98"], 4), "centre", round(o["centre"]["planned"]["TER_D98"], 4), flush=True)
        json.dump(res, open("../data/het_gnr.json", "w"), indent=1, default=float)
    print("done", round(time.time() - t0))
