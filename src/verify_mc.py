"""MC verification against the MCML slab benchmark (Wang et al. 1995): mua=1, mus=9 mm^-1, g=0.75, d=0.2 mm, matched boundaries."""
import os, sys, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import mc
geo = mc.Cylinder(R=1000.0, H=0.2, layers=[(0.0, 0.2, 1.0, 9.0, 0.75, 1.0)], n_out=1.0)
o = mc.run_cylinder(geo, beam_radius=1e-6, P=1.0, N=400000, nr=10, nz=20, seed=3)
res = dict(Rd=float(o["R_top"]), Tt=float(o["T_bot"]), Rd_ref=0.09734, Tt_ref=0.66096); print(res)
json.dump(res, open("../data/verify_mc.json", "w"), indent=1)
