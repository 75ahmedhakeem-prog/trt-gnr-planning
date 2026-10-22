import pandas as pd, numpy as np, json
d = pd.read_csv("../data/trt_dataset.csv")
s = d[np.isclose(d.perf, 1e-3) & np.isclose(d.tint_h, 0.5)]
ok = (s.T_skin_max <= 43) & (s.TER_normal_D2 <= 1.02) & (s.T_tum_max <= 45)
out = {}
for g in sorted(s.mua_gnr.unique()):
    b = s[ok & np.isclose(s.mua_gnr, g)].groupby(["lam_nm", "depth_mm", "Dt_mm"]).TER_D98.max().unstack("lam_nm")
    b = b.reindex(pd.MultiIndex.from_product([sorted(s.depth_mm.unique()), sorted(s.Dt_mm.unique())])).fillna(1.0)
    out[g] = {int(l): dict(frac_ge_1p10=float((b[l] >= 1.10).mean()), median=float(b[l].median()), max=float(b[l].max())) for l in (808, 1064)}
    print(g, out[g])
json.dump(out, open("../data/trt_maps.json", "w"), indent=1)
b = s[ok & np.isclose(s.mua_gnr, 0.2)].groupby(["lam_nm", "depth_mm", "Dt_mm"]).TER_D98.max()
b.to_csv("../data/trt_bestTER_gnr0.2.csv")
