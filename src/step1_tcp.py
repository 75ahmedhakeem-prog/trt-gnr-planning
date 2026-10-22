import json, numpy as np, coupled as C
m = np.load("../data/step1_maps.npz"); et, Tm = m["808_nm_MPE_S25_h200"]
_, tum, skin = C.session(808, C.MPE[808], 25, 2.5, 9.0, 0.15, 1e-3, 200.0, t_ht=60.0)
Vt = C.VC[tum]; e = et[tum]; p = C.TUM
doses = np.arange(20, 80.1, 2.0); out = {"dose": doses.tolist(), "RT": [], "RT+HT": []}
for Dtot in doses:
    n = Dtot / 2.0
    out["RT"].append(C.tcp(np.full(e.shape, Dtot), Vt, p))
    out["RT+HT"].append(C.tcp((n - 5) * 2.0 + 5 * e, Vt, p))
def dq(y, q): return float(np.interp(q, y, doses))
for q in (0.5, 0.9):
    out[f"D{int(q*100)}_RT"] = dq(np.array(out["RT"]), q); out[f"D{int(q*100)}_RTHT"] = dq(np.array(out["RT+HT"]), q)
print({k: (round(v, 2) if isinstance(v, float) else None) for k, v in out.items() if k.startswith("D")})
i46 = list(doses).index(46.0); print("TCP at 46 Gy: RT", round(out["RT"][i46], 3), "RT+HT", round(out["RT+HT"][i46], 3), " tumour volume mm3", round(Vt.sum(), 1))
json.dump(out, open("../data/step1_tcp.json", "w"), indent=1)
