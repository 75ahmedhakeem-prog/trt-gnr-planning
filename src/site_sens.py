"""Superficial-site sensitivity of the reference case (reviewer comment on cervical/pelvic reference data).
The temperature history of the reference case is kept; the tumour and normal-tissue LQ parameters and the schedule
are changed to values representative of superficial sites treated with thermoradiotherapy.
  Tumour sets: SiHa (reference), HeLa, SiHa thermal ratios with site alpha/beta (head-and-neck SCC 10 Gy,
  breast 3.5 Gy, melanoma 2.5 Gy), and the in vivo C3H decay constant (0.35 1/h, superficial murine tumour).
  Variants with beta41/beta37 = 1 (heat does not lower beta) bound the effect of the SiHa beta ratio at low alpha/beta.
  Normal tissue: late skin/subcutis (alpha/beta 3 Gy) and acute skin (alpha/beta 10 Gy).
  Schedules: 23 x 2 Gy with five weekly heating sessions (reference) and 8 x 4 Gy twice weekly with weekly heating
  (four sessions), the re-irradiation schedule for recurrent breast cancer (Linthorst et al. 2015).
TER = D98 of the tumour EQD2 / EQD2 of radiotherapy alone (same as Eq. (10) for 2 Gy fractions)."""
import os, sys, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import coupled as C
from trbio import PAR

TUMS = {"SiHa (reference)": dict(C.TUM), "HeLa": dict(PAR["HeLa"], mu=0.5),
        "Head-and-neck SCC (a/b 10 Gy)": dict(C.TUM, ab37=10.0), "Breast (a/b 3.5 Gy)": dict(C.TUM, ab37=3.5),
        "Melanoma (a/b 2.5 Gy)": dict(C.TUM, ab37=2.5),
        "Breast (a/b 3.5 Gy), b41/b37 = 1": dict(C.TUM, ab37=3.5, rb=1.0), "Melanoma (a/b 2.5 Gy), b41/b37 = 1": dict(C.TUM, ab37=2.5, rb=1.0),
        "Head-and-neck SCC (a/b 10 Gy), b41/b37 = 1": dict(C.TUM, ab37=10.0, rb=1.0), "SiHa, in vivo C3H decay (mu 0.35 1/h)": dict(C.TUM, mu=0.35)}
NRMS = {"late (a/b 3 Gy)": dict(C.NRM), "acute skin (a/b 10 Gy)": dict(C.NRM, ab37=10.0)}
SCHED = {"23 x 2 Gy, 5 HT": (23, 2.0, 5), "8 x 4 Gy, 4 HT": (8, 4.0, 4)}

def run(Th, tum, skin, tint=0.5):
    fat = ~tum & ~skin; out = []
    for sn, (n, d, nh) in SCHED.items():
        for tn, pt in TUMS.items():
            b = pt["a37"] / pt["ab37"]; e0 = (pt["a37"] * d + b * d * d) / (pt["a37"] + b * 2.0)      # EQD2 of an unheated fraction
            et, _ = C.fraction_eqd(Th, pt, d=d, tint=tint); tot = (n - nh) * e0 + nh * et; Vt = C.VC[tum]
            ter = C.pct(tot[tum], Vt, 0.02) / (n * e0)
            for nn, pn in NRMS.items():
                bn = pn["a37"] / pn["ab37"]; e0n = (pn["a37"] * d + bn * d * d) / (pn["a37"] + bn * 2.0)
                en, _ = C.fraction_eqd(Th, pn, d=d, tint=tint, direct=False); totn = (n - nh) * e0n + nh * en
                tern = max(C.pct(totn[skin], C.VC[skin], 0.98), C.pct(totn[fat], C.VC[fat], 0.98)) / (n * e0n)
                out.append(dict(schedule=sn, tumour=tn, normal=nn, EQD2_RT=n * e0, TER=ter, gain_Gy=(ter - 1) * n * e0, TER_n=tern, TGF=ter / tern))
    return out

if __name__ == "__main__":
    Th, tum, skin = C.session(808, C.MPE[808], 25, 2.5, 9.0, 0.15, 1e-3, 200.0)
    res = run(Th, tum, skin)
    for r in res: print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    json.dump(res, open("../data/site_sens.json", "w"), indent=1)
