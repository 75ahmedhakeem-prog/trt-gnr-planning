from plotstyle import *
plt.rcParams.update({"font.size": 16.5*1.35, "axes.labelsize": 18*1.35, "legend.fontsize": 13.5*1.35, "xtick.major.size": 8, "ytick.major.size": 8})
_lab = lab
def lab(ax, s, x=0.04, y=0.95, ha="left", **k): ax.text(x, y, s, transform=ax.transAxes, ha=ha, va="top", fontsize=19*1.35, fontweight="bold", **k)
import json, pandas as pd
f = pd.read_csv("../data/design_trt_feasible.csv"); r = json.load(open("../data/design_trt.json"))
fig, axs = plt.subplots(1, 2, figsize=(13, 4.9), gridspec_kw=dict(wspace=0.3))
a = axs[0]
for lam, col, mk in ((808, BLUE, "o"), (1064, RED, "s")):
    s = f[(f.lam_nm == lam) & np.isclose(f.tint_h, 0.5)].groupby("f_mpe").TER_D98.max()
    a.plot(s.index, s.values, "-", color=col, lw=2.4, label=f"{lam} nm (ANN)")
    for key, fill in ((f"{lam}|max_TER", col), (f"{lam}|min_power_97pct", "white")):
        d = r[key]; a.plot(d["f_mpe"], d["dP1"]["TER_D98"], mk, ms=11, mfc=fill, mec=col, mew=2)
a.set_xlabel("Irradiance / MPE"); a.set_ylabel("Tumour TER (D$_{98}$)"); a.set_xlim(0.1, 1.02); a.set_ylim(1.0, 1.20)
a.legend(loc="upper left", fontsize=18); a.xaxis.set_minor_locator(AutoMinorLocator(2)); a.yaxis.set_minor_locator(AutoMinorLocator(2)); lab(a, "(a)", x=0.96, ha="right", y=0.2)
b = axs[1]; keys = ["808|max_TER", "808|min_power_97pct", "1064|max_TER", "1064|min_power_97pct"]
lbl = ["808 nm\nmax", "808 nm\nlow power", "1064 nm\nmax", "1064 nm\nlow power"]; x = np.arange(4); w = 0.26
for off, src, col, name in ((-w, None, GREEN, "ANN"), (0, "dP1", ORANGE, "δP1"), (w, "MC", PURPLE, "MC")):
    v = [r[k]["TER_D98"] if src is None else r[k][src]["TER_D98"] for k in keys]
    b.bar(x + off, np.array(v) - 1, w, color=col, ec=BLACK, lw=1.1, label=name)
b.set_xticks(x); b.set_xticklabels(lbl, fontsize=17.6); b.tick_params(axis="x", which="both", top=False, length=0)
b.set_ylabel("Tumour TER − 1"); b.set_ylim(0, 0.18); b.legend(loc="upper left", ncol=3, fontsize=18, columnspacing=0.8, handlelength=1.5); b.yaxis.set_minor_locator(AutoMinorLocator(2)); lab(b, "(b)", x=0.96, ha="right", y=0.80)
fig.savefig("../figures/trt_step4c_design.png"); plt.close(fig); print("ok")
