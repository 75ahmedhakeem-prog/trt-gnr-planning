"""In vivo test of the decay of thermal radiosensitisation with the heat-radiation interval.
Data: Overgaard, Int J Radiat Oncol Biol Phys 6 (1980) 1507, Table 2 (C3H mammary carcinoma and overlying skin,
42.5 C for 60 min, single doses, TER from TCD50 / DD50).  In the extended LQ model ln(TER) - ln(TER_inf) is
proportional to exp(-mu t) when one LQ term dominates, so ln TER(t) = ln TER_inf + K exp(-mu t) is fitted by
weighted least squares (weights from the 95% confidence limits of the combined-treatment TCD50 / DD50).
The fitted decay constants are then used in the reference case."""
import json, numpy as np
from scipy.optimize import curve_fit
import gsa
t = np.array([0, 0.5, 1, 4, 8, 24.0])
# TER and 95% CI of the combined-treatment dose (rad) from Table 2
D = {"tumour, heat before RT": ([2.45, 2.20, 2.18, 1.69, 1.51, 1.52], [(1913, 2750), (2262, 2887), (2315, 2862), (2901, 3785), (3415, 4014), (3330, 4070)], [2299, 2556, 2574, 3314, 3704, 3681]),
     "tumour, RT before heat": ([2.45, 2.13, 2.06, 1.52, 1.45, 1.59], [(1913, 2750), (2333, 2978), (2484, 2978), (2888, 4712), (3606, 4151), (3110, 3974)], [2299, 2635, 2720, 3692, 3866, 3516]),
     "skin, heat before RT": ([2.52, 1.96, 1.93, 1.27, 1.44, 1.10], [(947, 1140), (1065, 1743), (736, 2592), (1790, 2454), (1541, 2211), (2185, 2700)], [1054, 1363, 1381, 2095, 1847, 2429]),
     "skin, RT before heat": ([2.52, 1.86, 1.53, 0.91, 0.87, 0.98], [(947, 1140), (1181, 1742), (1486, 2047), (2631, 3266), (2503, 3719), (2419, 3051)], [1054, 1434, 1744, 2931, 3051, 2717])}
f = lambda t, K, mu, P: P + K * np.exp(-mu * t)
out = {"fits": {}}
for k, (ter, ci, d) in D.items():
    y = np.log(ter); s = np.array([np.log(hi / lo) / 3.92 for lo, hi in ci])         # SE of ln(dose) from the 95% CI
    po, pc = curve_fit(f, t, y, p0=[0.5, 0.5, 0.3], sigma=s, absolute_sigma=True, bounds=([0, 0.01, -1], [3, 10, 2]))
    se = np.sqrt(np.diag(pc)); r = y - f(t, *po)
    out["fits"][k] = dict(K=po[0], mu=po[1], mu_se=se[1], TER_inf=float(np.exp(po[2])), rms_lnTER=float(np.sqrt((r**2).mean())), t=t.tolist(), TER=ter)
    print(f"{k:24s} mu = {po[1]:.2f} +/- {se[1]:.2f} 1/h, TER_inf = {np.exp(po[2]):.2f}, rms = {np.sqrt((r**2).mean()):.3f}")
mt = out["fits"]["tumour, heat before RT"]["mu"]; mn = out["fits"]["skin, RT before heat"]["mu"]; mn2 = out["fits"]["skin, heat before RT"]["mu"]
out["reference_with_in_vivo_mu"] = {}
for a, b in ((0.5, 1.0), (mt, 1.0), (0.5, mn), (mt, mn), (mt, mn2)):
    r = gsa.run(dict(gsa.REF, mu=a, mu_n=b)); key = f"mu={a:.2f}, mu_n={b:.2f}"
    out["reference_with_in_vivo_mu"][key] = {q: r[q] for q in ("TER", "TER_n", "TGF", "dD50")}
    print(key, {q: round(r[q], 3) for q in ("TER", "TER_n", "TGF", "dD50")})
json.dump(out, open("../data/bench_overgaard.json", "w"), indent=1, default=float)
