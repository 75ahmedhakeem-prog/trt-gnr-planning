from plotstyle import *
plt.rcParams.update({"font.size": 16.5*1.35, "axes.labelsize": 18*1.35, "legend.fontsize": 13.5*1.35, "xtick.major.size": 8, "ytick.major.size": 8})
_lab = lab
def lab(ax, s, x=0.04, y=0.95, ha="left", **k): ax.text(x, y, s, transform=ax.transAxes, ha=ha, va="top", fontsize=19*1.35, fontweight="bold", **k)
import json, coupled as Cp, design_sim as ds
m = np.load("../data/step1_maps.npz"); et, Tm = m["808_nm_MPE_S25_h200"]; tc = json.load(open("../data/step1_tcp.json"))
tum, skin = ds.regions(ds.rt, ds.zt, 2.5, 9.0)
fig, axs = plt.subplots(1, 3, figsize=(16, 5.6), gridspec_kw=dict(width_ratios=[1, 1, 1.05], wspace=0.55))
ext = [-ds.rt[-1], ds.rt[-1], ds.zt[-1], 0]
for a, F, lb, cm, vr, l in ((axs[0], Tm, "Mean T (°C)", "inferno", (37, 44), "(a)"), (axs[1], et, "EQD per fraction (Gy)", "viridis", (2, 5), "(b)")):
    full = np.vstack([F[::-1], F]).T
    im = a.imshow(full, extent=ext, cmap=cm, vmin=vr[0], vmax=vr[1], aspect="auto")
    tf = np.vstack([tum[::-1], tum]).T.astype(float); a.contour(np.linspace(ext[0], ext[1], tf.shape[1]), ds.zt, tf, [0.5], colors="white", linewidths=2)
    a.axhline(1.5, color="white", ls="--", lw=1.5); a.set_xlim(-15, 15); a.set_ylim(16, 0)
    a.set_xlabel("r (mm)"); a.set_ylabel("Depth (mm)"); cb = fig.colorbar(im, ax=a, fraction=0.05, pad=0.03); cb.set_label(lb, fontsize=20.2)
    a.text(0.04, 0.04, l, transform=a.transAxes, ha="left", va="bottom", fontsize=25.7, fontweight="bold", color="white")
a = axs[2]; d = np.array(tc["dose"])
a.plot(d, tc["RT"], "-", color=BLUE, lw=2.6, label="RT alone"); a.plot(d, tc["RT+HT"], "-", color=RED, lw=2.6, label="RT + 5 HT")
for q, ls in ((0.5, ":"), (0.9, "--")): a.axhline(q, color=GREY, ls=ls, lw=1.3)
a.annotate("", xy=(tc["D50_RTHT"], 0.5), xytext=(tc["D50_RT"], 0.5), arrowprops=dict(arrowstyle="<->", color=BLACK, lw=1.8))
a.text(tc["D50_RT"] + 2.5, 0.56, f"−{tc['D50_RT'] - tc['D50_RTHT']:.0f} Gy", ha="left", va="bottom", fontsize=20.2)
a.set_xlabel("Total dose (Gy)"); a.set_ylabel("TCP"); a.set_xlim(20, 70); a.set_ylim(0, 1.05)
a.legend(loc="lower right", fontsize=17.6, bbox_to_anchor=(1.0, 0.04)); a.xaxis.set_minor_locator(AutoMinorLocator(2)); a.yaxis.set_minor_locator(AutoMinorLocator(2)); lab(a, "(c)")
fig.savefig("../figures/trt_step1.png"); plt.close(fig); print("ok")
