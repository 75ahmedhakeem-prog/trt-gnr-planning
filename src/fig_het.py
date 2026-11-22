"""Figure: heterogeneous GNR distributions at equal total gold (het_gnr.py)."""
from plotstyle import *
import json, matplotlib.gridspec as gs
import het_gnr as H, design_sim as ds
R = json.load(open("../data/het_gnr.json"))
PAT = [("uniform", "Uniform"), ("rim", "Rim"), ("shell", "Necrotic core"), ("centre", "Central"), ("light-side", "Light side"), ("far-side", "Far side"),
       ("lognormal s1.0 l0.5 fixed", "Random, σ 1, 0.5 mm"), ("lognormal s1.0 l2.0 fixed", "Random, σ 1, 2 mm"), ("lognormal s1.0 l2.0 varying", "Random, σ 1, 2 mm, varying"),
       ("washout 15%/session", "Wash-out")]
CASES = [("Reference case", BLACK, "o"), ("808 nm max. TER", BLUE, "s"), ("808 nm low power", GREEN, "D"), ("1064 nm max. TER", RED, "^"), ("1064 nm low power", ORANGE, "v")]
fig = plt.figure(figsize=(16, 11.5)); G = gs.GridSpec(2, 7, height_ratios=[1, 1.35], hspace=0.5, wspace=0.12)
maps = [("uniform", "Uniform"), ("rim", "Rim"), ("shell", "Necrotic core"), ("centre", "Central"), ("light-side", "Light side"), ("far-side", "Far side"), ("lognormal", "Random")]
ir = ds.rc <= 6.0; iz = (ds.zc >= 2.5) & (ds.zc <= 14.5)
for j, (k, t) in enumerate(maps):
    a = fig.add_subplot(G[0, j]); c = H.pattern(k, 2.5, 9.0, seed=3, sig=1.0, ell=0.5)
    m = np.ma.masked_where(c[np.ix_(ir, iz)].T <= 0, c[np.ix_(ir, iz)].T)
    im = a.imshow(m, extent=[0, 6, 14.5, 2.5], vmin=0, vmax=3, cmap="viridis", aspect="auto")
    a.add_patch(matplotlib.patches.Arc((0, 1.5 + 2.5 + 4.5), 9, 9, theta1=-90, theta2=90, color=BLACK, lw=1.5))
    a.set_title(t, fontsize=15); a.set_xticks([0, 3, 6]); a.tick_params(labelsize=13)
    if j: a.set_yticklabels([])
    else: a.set_ylabel("z (mm)"); lab(a, "(a)", x=-0.75, y=1.25)
    a.set_xlabel("r (mm)", fontsize=14)
cb = fig.colorbar(im, ax=fig.axes[:7], fraction=0.015, pad=0.01); cb.set_label("c / c$_{mean}$", fontsize=15); cb.ax.tick_params(labelsize=13)
def val(case, k, q, which="planned"):
    v = R[case][k]
    if isinstance(v, list): x = np.array([r[which][q] for r in v]); return np.median(x), x.min(), x.max()
    x = v[which][q]; return x, x, x
for col, (q, yl, l) in enumerate((("TER_D98", "Tumour TER (D$_{98}$)", "(b)"), ("Ttum_max", "T$_{tum,max}$ (°C)", "(c)"))):
    a = fig.add_subplot(G[1, col * 4: col * 4 + 3 + (col == 0)]) if False else fig.add_subplot(G[1, 0:4] if col == 0 else G[1, 4:7])
    x = np.arange(len(PAT))
    for i, (case, colr, mk) in enumerate(CASES):
        v = np.array([val(case, k, q) for k, _ in PAT]); dx = (i - 2) * 0.13
        a.errorbar(x + dx, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt=mk, color=colr, ms=8, capsize=3, lw=1.5, label=case)
    if q == "Ttum_max": a.axhline(45, color=GREY, ls="--", lw=1.6)
    a.set_xticks(x); a.set_xticklabels([t for _, t in PAT], rotation=40, ha="right", fontsize=14); a.tick_params(axis="x", which="both", top=False)
    a.set_ylabel(yl); a.yaxis.set_minor_locator(AutoMinorLocator(2)); lab(a, l, x=0.02, y=0.97)
    if col == 0: hh, ll = a.get_legend_handles_labels()
fig.legend(hh, ll, loc="center", bbox_to_anchor=(0.5, 0.525), ncol=5, fontsize=14, handletextpad=0.3, columnspacing=1.2)
fig.savefig("../figures/trt_het.png"); print("ok")
