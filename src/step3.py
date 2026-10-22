"""Dual role of gold: photothermal radiosensitisation versus physical dose enhancement, as a function of gold concentration."""
import json, numpy as np, coupled as C
from trbio import lq, k_heat
D = json.load(open("../data/step3_def.json")); MUA_PER = np.sqrt(0.0202 * 0.0589)          # mm^-1 per ug/mL at 808 nm (geometric mean of damping bounds)
N_FX, N_HT = 23, 5
def ter_total(c_ugg, beam, heat=True):
    mua = c_ugg * MUA_PER; w = c_ugg * 1e-6; DEF = 1 + w * (D[beam]["R"] - 1)
    Th, tum, skin = C.session(808, C.MPE[808], 25, 2.5, 9.0, mua, 1e-3, 200.0)
    if not heat: Th = np.full_like(Th, 37.0)
    p = C.TUM; d = 2.0 * DEF; Tm = Th.mean(0)
    a, b = lq(Tm, 0.5, p); a37 = p["a37"]; b37 = a37 / p["ab37"]; den = a37 + b37 * 2
    e_ht = (a * d + b * d**2) / den + k_heat(Th, p).sum(0) * 30.0 / den
    e_rt = (a37 * d + b37 * d**2) / den
    tot = (N_FX - N_HT) * e_rt + N_HT * e_ht
    Vt = C.VC[tum]; return C.pct(tot[tum], Vt, 0.02) / (N_FX * 2.0), C.pct(Tm[tum], Vt, 0.5), C.pct(Tm[tum], Vt, 0.1), float(Th[:, skin].max()), mua
res = {}
for beam in ("100 kVp (2 mm Al)", "250 kVp (1 mm Cu)", "6 MV + 10% low-energy scatter"):
    res[beam] = {}
    for c in (1, 3, 5, 10, 30, 100, 300, 1000, 3000, 10000):
        t_h, T50, T90, Ts, mua = ter_total(c, beam, True); t_n, *_ = ter_total(c, beam, False)
        res[beam][c] = dict(TER_heat_plus_DEF=t_h, TER_DEF_only=t_n, T50=T50, T90=T90, Tskin=Ts, mua=mua)
        print(beam, c, {k: round(v, 3) for k, v in res[beam][c].items()}, flush=True)
json.dump(res, open("../data/step3.json", "w"), indent=1)
