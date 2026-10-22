"""Timing: DNA-repair inhibition (decays with mu) versus cell-cycle redistribution (peaks ~24 h after heating)."""
import json, numpy as np, coupled as C
from trbio import lq, k_heat
from scipy.optimize import brentq
Th, tum, skin = C.session(808, C.MPE[808], 25, 2.5, 9.0, 0.15, 1e-3, 200.0)
Vt = C.VC[tum]; Tm = Th.mean(0); KD = (k_heat(Th, C.TUM).sum(0) * 30.0)
def g(t, tp=24.0): t = np.maximum(t, 0); return (t / tp) * np.exp(1 - t / tp)          # redistribution time course, g(tp) = 1
def eqd_fx(t, A, p=C.TUM, d=2.0, dref=2.0, T=None):
    T = Tm if T is None else T
    a, b = lq(T, t, p); a37 = p["a37"]; b37 = a37 / p["ab37"]; den = a37 + b37 * dref
    a = a * (1 + A * g(t))
    return (a * d + b * d**2) / den + KD / den
def d98(e): return C.pct(e[tum], Vt, 0.02)
res = {}
for mu in (0.3, 0.5, 1.0):
    p = dict(C.TUM, mu=mu)
    e05 = d98(eqd_fx(0.5, 0, p)); e24 = d98(eqd_fx(24.0, 0, p))
    Astar = brentq(lambda A: d98(eqd_fx(24.0, A, p)) - e05, 0, 5)
    res[f"mu={mu}"] = dict(EQD_0p5h=e05, EQD_24h_noRedistribution=e24, A_breakeven=Astar)
# break-even A* versus uniform tumour temperature (mu = 0.5)
res["A_vs_T"] = {}
for T in (40.0, 41.0, 42.0, 43.0):
    TT = np.full(Tm.shape, T); KDs = k_heat(T, C.TUM) * 3600.0
    def e(t, A):
        a, b = lq(TT, t, C.TUM); a37 = C.TUM["a37"]; b37 = a37 / C.TUM["ab37"]; den = a37 + b37 * 2
        return float(((a * (1 + A * g(t)) * 2 + b * 4) / den + KDs / den)[0, 0])
    res["A_vs_T"][T] = brentq(lambda A: e(24.0, A) - e(0.5, 0), 0, 10)
# interpretation: alpha increase from a G2/M fraction shift f0 -> f1 with G2/M relative sensitivity r
def A_from_cycle(f0, f1, r): return ((f1 * r + (1 - f1)) / (f0 * r + (1 - f0))) - 1
res["A_cycle_examples"] = {f"f0={f0},f1={f1},r={r}": A_from_cycle(f0, f1, r) for f0 in (0.15,) for f1 in (0.3, 0.45, 0.6) for r in (1.5, 2.0, 3.0)}
# daily fractionation: heating 30 min before a fraction; the next fraction follows ~24 h later
res["daily_schedule_gain_per_session"] = {}
for A in (0.0, 0.1, 0.2, 0.3):
    first = d98(eqd_fx(0.5, 0)) - 2.0; nxt = d98(eqd_fx(24.0, A) - KD / (C.TUM["a37"] + C.TUM["a37"] / C.TUM["ab37"] * 2)) - 2.0
    res["daily_schedule_gain_per_session"][A] = dict(first_fraction=first, next_day_fraction=nxt, total=first + nxt)
json.dump(res, open("../data/step2.json", "w"), indent=1, default=float)
print(json.dumps(res, indent=1, default=lambda x: round(float(x), 3)))
