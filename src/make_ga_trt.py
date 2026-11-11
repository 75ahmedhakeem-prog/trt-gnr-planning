from plotstyle import *
plt.rcParams.update({"font.size": 16.5*1.35, "axes.labelsize": 18*1.35, "legend.fontsize": 13.5*1.35, "xtick.major.size": 8, "ytick.major.size": 8})
_lab = lab
def lab(ax, s, x=0.04, y=0.95, ha="left", **k): ax.text(x, y, s, transform=ax.transAxes, ha=ha, va="top", fontsize=19*1.35, fontweight="bold", **k)
import json, design_sim as ds
from matplotlib.patches import FancyArrowPatch
plt.rcParams.update({"font.size": 25.7, "axes.labelsize": 27.0, "xtick.labelsize": 24.3, "ytick.labelsize": 24.3, "axes.linewidth": 2.2})
m = np.load("../data/step1_maps.npz"); et, Tm = m["808_nm_MPE_S25_h200"]; tc = json.load(open("../data/step1_tcp.json")); g = json.load(open("../data/step3.json"))
tum, skin = ds.regions(ds.rt, ds.zt, 2.5, 9.0)
fig = plt.figure(figsize=(16, 6.2)); TT = dict(ha="center", fontsize=29.7, fontweight="bold")
a = fig.add_axes([0.045, 0.17, 0.22, 0.66]); full = np.vstack([Tm[::-1], Tm]).T
im = a.imshow(full, extent=[-ds.rt[-1], ds.rt[-1], ds.zt[-1], 0], cmap="inferno", vmin=37, vmax=44, aspect="auto")
tf = np.vstack([tum[::-1], tum]).T.astype(float); a.contour(np.linspace(-ds.rt[-1], ds.rt[-1], tf.shape[1]), ds.zt, tf, [0.5], colors="white", linewidths=2.5)
a.set_xlim(-14, 14); a.set_ylim(15, 0); a.set_xlabel("r (mm)"); a.set_ylabel("Depth (mm)")
cb = fig.colorbar(im, ax=a, fraction=0.06, pad=0.03); cb.set_label("T (°C)", fontsize=25.7)
a.text(0.05, 0.05, "(a)", transform=a.transAxes, fontsize=31.1, fontweight="bold", color="white")
fig.text(0.16, 0.92, "808 nm at the MPE: 41–43 °C", **TT)
b = fig.add_axes([0.41, 0.17, 0.24, 0.66]); d = np.array(tc["dose"])
b.plot(d, tc["RT"], "-", color=BLUE, lw=3, label="RT alone"); b.plot(d, tc["RT+HT"], "-", color=RED, lw=3, label="RT + GNR heat")
b.annotate("", xy=(tc["D50_RTHT"], 0.5), xytext=(tc["D50_RT"], 0.5), arrowprops=dict(arrowstyle="<->", lw=2.2))
b.text(47.5, 0.5, "−10 Gy", va="center", ha="left", fontsize=27.0, fontweight="bold")
b.text(46.6, 0.355, "(range 1–19 Gy)", va="center", ha="left", fontsize=16.5, fontweight="bold")
b.set_xlim(20, 70); b.set_ylim(0, 1.05); b.set_xlabel("Radiation dose (Gy)"); b.set_ylabel("TCP"); b.legend(loc="lower right", bbox_to_anchor=(1.0, 0.0), fontsize=15, handlelength=0.8, handletextpad=0.4, borderaxespad=0.2)
b.text(0.05, 0.95, "(b)", transform=b.transAxes, fontsize=31.1, fontweight="bold", va="top")
fig.text(0.53, 0.92, "Same control, less radiation", **TT)
c = fig.add_axes([0.745, 0.17, 0.24, 0.66])
for beam, col, nm in (("100 kVp (2 mm Al)", RED, "100 kVp"), ("6 MV + 10% low-energy scatter", BLUE, "6 MV")):
    cc = np.array([float(k) for k in g[beam]]); c.plot(cc, [v["TER_heat_plus_DEF"] for v in g[beam].values()], "-o", color=col, lw=2.6, ms=7, label=f"{nm}: heat + gold")
    c.plot(cc, [v["TER_DEF_only"] for v in g[beam].values()], "--", color=col, lw=2.4, label=f"{nm}: gold only")
c.set_xscale("log"); c.set_ylim(0.95, 2.0); c.set_xlabel("Gold (μg g$^{-1}$)"); c.set_ylabel("Tumour TER"); c.legend(loc="upper left", fontsize=17)
c.text(0.55, 0.55, "(c)", transform=c.transAxes, fontsize=31.1, fontweight="bold", ha="center")
fig.text(0.865, 0.92, "Gold works as a heater", **TT)
fig.savefig("../figures/trt_ga.png", dpi=600); plt.close(fig); print("ok")
