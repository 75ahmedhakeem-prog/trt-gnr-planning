from plotstyle import *
plt.rcParams.update({"font.size": 16.5*1.35, "axes.labelsize": 18*1.35, "legend.fontsize": 13.5*1.35, "xtick.major.size": 8, "ytick.major.size": 8})
_lab = lab
def lab(ax, s, x=0.04, y=0.95, ha="left", **k): ax.text(x, y, s, transform=ax.transAxes, ha=ha, va="top", fontsize=19*1.35, fontweight="bold", **k)
import pandas as pd
h = pd.read_csv("../data/ml_trt_holdout_v2.csv")
names = {"spot_mm": "Spot", "depth_mm": "Depth", "Dt_mm": "Diameter", "perf": "Perfusion", "mua_gnr": "μ$_{a,GNR}$", "f_mpe": "I/MPE", "tint_h": "Interval"}
fig, axs = plt.subplots(1, 2, figsize=(13, 4.9))
for a, tg, l, ttl in ((axs[0], "TER_D98", "(a)", "Tumour TER (D$_{98}$)"), (axs[1], "T_skin_max", "(b)", "Maximum skin temperature")):
    s = h[h.target == tg].groupby("variable")[["ANN_R2", "GBT_R2"]].mean().reindex(list(names)); x = np.arange(len(s)); w = 0.38
    a.bar(x - w / 2, s.GBT_R2.clip(lower=0), w, color=GREEN, ec=BLACK, lw=1.1, label="GBT")
    a.bar(x + w / 2, s.ANN_R2, w, color=RED, ec=BLACK, lw=1.1, label="ANN")
    a.set_xticks(x); a.set_xticklabels([names[v] for v in s.index], rotation=35, ha="right", fontsize=17.6); a.tick_params(axis="x", which="both", top=False, length=0)
    a.set_ylim(0, 1.4); a.set_ylabel("Level hold-out R$^2$"); a.legend(loc="upper left", ncol=2, fontsize=18); a.set_title(ttl, fontsize=19.6, fontweight="bold")
    a.text(0.97, 0.96, l, transform=a.transAxes, ha="right", va="top", fontsize=25.7, fontweight="bold"); a.yaxis.set_minor_locator(AutoMinorLocator(2))
fig.tight_layout(); fig.savefig("../figures/trt_step4b_ml.png"); plt.close(fig); print("ok")
