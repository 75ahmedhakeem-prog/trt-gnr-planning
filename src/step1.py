import json, numpy as np
import coupled as C
from trbio import PAR
REF = dict(depth=2.5, Dt=9.0, gnr=0.15, perf=1e-3)
SETS = [  # SiHa 14 sigma-point sets (van Leeuwen 2017, Table B2): a37, ab, ra, rb, (mu), dS ; mu fixed at clinical 0.5
 (0.344,10.571,1.860,0.431,392.155),(0.429,25.256,1.608,0.404,392.009),(0.387,17.907,1.822,0.258,391.293),(0.386,17.920,1.646,0.577,392.872),
 (0.388,17.915,1.801,0.175,392.139),(0.385,17.912,1.667,0.660,392.026),(0.381,17.915,1.794,0.435,392.085),(0.391,17.912,1.674,0.400,392.079),
 (0.392,17.913,1.730,0.416,392.082),(0.381,17.914,1.738,0.419,392.083),(0.386,17.914,1.734,0.418,392.082),(0.386,17.914,1.734,0.418,392.082),
 (0.392,17.914,1.735,0.418,392.082),(0.381,17.914,1.733,0.417,392.082)]
N_FX, N_HT, D = 23, 5, 2.0
def evaluate(Th, tum, skin, tint=0.5, p_t=C.TUM, p_n=C.NRM):
    fat = ~tum & ~skin
    et, Tm = C.fraction_eqd(Th, p_t, tint=tint); en, _ = C.fraction_eqd(Th, p_n, tint=tint, direct=False)
    Vt, Vs, Vf = C.VC[tum], C.VC[skin], C.VC[fat]
    tot_t = (N_FX - N_HT) * D + N_HT * et[tum]; tot_n_s = (N_FX - N_HT) * D + N_HT * en[skin]; tot_n_f = (N_FX - N_HT) * D + N_HT * en[fat]
    ter_t = C.pct(tot_t, Vt, 0.02) / (N_FX * D)                      # D98 of tumour
    ter_s = C.pct(tot_n_s, Vs, 0.98) / (N_FX * D); ter_f = C.pct(tot_n_f, Vf, 0.98) / (N_FX * D)   # D2 of skin / fat
    return dict(T90=C.pct(Tm[tum], Vt, 0.10), T50=C.pct(Tm[tum], Vt, 0.5), Tmax_tum=float(Th[:, tum].max()), Tskin_max=float(Th[:, skin].max()),
                Tfat_max=float(Th[:, fat].max()), EQDfx_D98=C.pct(et[tum], Vt, 0.02), TER_D98=ter_t, TER_skin_D2=ter_s, TER_fat_D2=ter_f,
                TGF=ter_t / max(ter_s, ter_f), et=et, Tm=Tm)
res = {}; maps = {}
cases = [("808 nm, MPE, S25, h200", 808, None, 25, 200.0), ("808 nm, MPE, S25, h1000", 808, None, 25, 1000.0),
         ("808 nm, MPE, S20, h200", 808, None, 20, 200.0)]
for name, lam, I, S, h in cases:
    I = C.MPE[lam] if I is None else I
    Th, tum, skin = C.session(lam, I, S, REF["depth"], REF["Dt"], REF["gnr"], REF["perf"], h)
    r = evaluate(Th, tum, skin); maps[name] = (r.pop("et"), r.pop("Tm"), tum, skin); r["I"] = I; res[name] = r
    print(name, {k: round(v, 3) for k, v in r.items()}, flush=True)
# 1064 nm scaled to give the same tumour T50 as the 808-nm reference case
Th1, tum, skin = C.session(1064, 1.0, 25, REF["depth"], REF["Dt"], REF["gnr"], REF["perf"], 200.0)
target = res["808 nm, MPE, S25, h200"]["T50"]; Tm1 = Th1.mean(0); I1 = (target - 37) / (C.pct(Tm1[tum], C.VC[tum], 0.5) - 37)
Th1 = 37 + I1 * (Th1 - 37); r = evaluate(Th1, tum, skin); maps["1064 nm, matched T50, S25, h200"] = (r.pop("et"), r.pop("Tm"), tum, skin); r["I"] = I1
res["1064 nm, matched T50, S25, h200"] = r; print("1064 matched", {k: round(v, 3) for k, v in r.items()})
# uncertainty of tumour TER over the 14 SiHa parameter sets, and HeLa as a second cell line (reference case)
Th, tum, skin = C.session(808, C.MPE[808], 25, REF["depth"], REF["Dt"], REF["gnr"], REF["perf"], 200.0)
ters = []
for a37, ab, ra, rb, dS in SETS:
    p = dict(C.TUM, a37=a37, ab37=ab, ra=ra, rb=rb, dS=dS); ters.append(evaluate(Th, tum, skin, p_t=p)["TER_D98"])
pH = dict(PAR["HeLa"]); pH["mu"] = 0.5; rH = evaluate(Th, tum, skin, p_t=pH)
res["uncertainty_SiHa_TER_D98"] = [round(min(ters), 4), round(max(ters), 4)]; res["HeLa_TER_D98"] = rH["TER_D98"]
# time-interval dependence of tumour TER and TGF (reference case)
res["tint"] = {}
for ti in (0.0, 0.5, 1.0, 2.0, 4.0):
    r = evaluate(Th, tum, skin, tint=ti); res["tint"][ti] = dict(TER=r["TER_D98"], TGF=r["TGF"])
print(res["uncertainty_SiHa_TER_D98"], res["HeLa_TER_D98"], res["tint"])
json.dump(res, open("../data/step1.json", "w"), indent=1, default=float)
np.savez("../data/step1_maps.npz", **{k.replace(" ", "_").replace(",", ""): np.stack([v[0], v[1]]) for k, v in maps.items()})
