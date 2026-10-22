"""Macroscopic dose-enhancement factor of gold (mixture rule with NIST mu_en/rho) for kV and MV beams."""
import numpy as np, pandas as pd, json
N = {k: pd.read_csv(f"../data/nist/{k}.csv") for k in ("au", "water", "al", "cu")}
def coef(m, E, col):
    t = N[m]; return np.exp(np.interp(np.log(E), np.log(t.E.values), np.log(t[col].values)))
def au_muen(E):   # handle edges: interpolate within the correct branch
    t = N["au"]; e = t.E.values; v = t.muen.values; out = np.empty_like(E)
    for i, x in enumerate(E):
        j = np.searchsorted(e, x) ; j = min(max(j, 1), len(e) - 1)
        out[i] = np.exp(np.interp(np.log(x), np.log(e[j-1:j+1]), np.log(v[j-1:j+1])))
    return out
def kramers(E0, filters, depth_cm=0.5, Emin=0.010, n=4000):
    E = np.linspace(Emin, E0, n)[:-1]; psi = (E0 - E)                       # Kramers energy fluence
    for m, t_cm, rho in filters: psi = psi * np.exp(-coef(m, E, "mu") * rho * t_cm)
    psi = psi * np.exp(-coef("water", E, "mu") * depth_cm); return E, psi
def ratio(E, psi):
    """dose-weighted mean of (mu_en/rho)_Au / (mu_en/rho)_water"""
    mw = coef("water", E, "muen"); return float(np.sum(psi * au_muen(E)) / np.sum(psi * mw))
beams = {"100 kVp (2 mm Al)": kramers(0.100, [("al", 0.2, 2.699)]),
         "250 kVp (1 mm Cu)": kramers(0.250, [("cu", 0.1, 8.96), ("al", 0.1, 2.699)]),
         "6 MV primary": kramers(6.0, [("cu", 1.0, 8.96)], Emin=0.1)}
E6, p6 = beams["6 MV primary"]; Es = np.linspace(0.05, 0.2, 300); ps = np.ones_like(Es)
w_sc = 0.10 * p6.sum() * (E6[1] - E6[0]) / (ps.sum() * (Es[1] - Es[0]))
res = {}
for k, (E, psi) in beams.items():
    R = ratio(E, psi); res[k] = dict(R=R, Emean_keV=float(np.sum(E * psi) / psi.sum() * 1e3))
E = np.concatenate([E6, Es]); psi = np.concatenate([p6, ps * w_sc]); res["6 MV + 10% low-energy scatter"] = dict(R=ratio(E, psi), Emean_keV=float(np.sum(E * psi) / psi.sum() * 1e3))
for k in res:
    R = res[k]["R"]; res[k]["DEF"] = {w: 1 + w * (R - 1) for w in (5e-6, 1e-4, 1e-3, 5e-3, 1e-2)}
    res[k]["w_for_5pct"] = 0.05 / (R - 1); res[k]["w_for_13pct"] = 0.134 / (R - 1)
json.dump(res, open("../data/step3_def.json", "w"), indent=1)
for k, v in res.items(): print(k, "R=%.1f Emean=%.0f keV" % (v["R"], v["Emean_keV"]), "DEF(5ppm)=%.5f DEF(0.1%%)=%.4f DEF(1%%)=%.3f" % (v["DEF"][5e-6], v["DEF"][1e-3], v["DEF"][1e-2]), " w for +13.4%%: %.2e (%.0f ug/g)" % (v["w_for_13pct"], v["w_for_13pct"] * 1e6))
