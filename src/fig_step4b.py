from plotstyle import *
plt.rcParams.update({"font.size": 16.5*1.35, "axes.labelsize": 18*1.35, "legend.fontsize": 13.5*1.35, "xtick.major.size": 8, "ytick.major.size": 8})
import pandas as pd
# level hold-out R2 of the ANN, GBT and the smooth-function baselines (same samples, ml_trt.py / ml_baselines.py)
h = pd.read_csv("../data/ml_trt_holdout_v2.csv")
# Spline columns of the first file are superseded by ml_baselines_holdout_Spline.csv (quantile knots)
for f in ("../data/ml_baselines_holdout_Poly2_Poly3_Poly4_Poly5_Spline_RBF.csv", "../data/ml_baselines_holdout_GP.csv", "../data/ml_baselines_holdout_Spline.csv"):
    b = pd.read_csv(f)
    if "GP" not in f and "_Spline.csv" not in f: b = b.drop(columns=[c for c in b.columns if c.startswith("Spline")])
    h = h.merge(b[[c for c in b.columns if c.endswith("_R2") or c in ("target", "variable", "level")]], on=["target", "variable", "level"])
names = {"spot_mm": "Spot", "depth_mm": "Depth", "Dt_mm": "Diameter", "perf": "Perfusion", "mua_gnr": "μ$_{a,GNR}$", "f_mpe": "I/MPE", "tint_h": "Interval"}
MOD = [("GBT", "GBT", GREEN), ("Poly4", "Poly-4", GREY), ("Spline", "Spline", PURPLE), ("GP", "GP", ORANGE), ("RBF", "RBF", BLUE), ("ANN", "ANN", RED)]
fig, axs = plt.subplots(2, 1, figsize=(14, 10.5))
for a, tg, l, ttl in ((axs[0], "TER_D98", "(a)", "Tumour TER (D$_{98}$)"), (axs[1], "T_skin_max", "(b)", "Maximum skin temperature")):
    s = h[h.target == tg].groupby("variable")[[m + "_R2" for m, _, _ in MOD]].mean().reindex(list(names)); x = np.arange(len(s)); w = 0.135
    for i, (m, nm, c) in enumerate(MOD):
        a.bar(x + (i - 2.5) * w, s[m + "_R2"].clip(lower=0), w, color=c, ec=BLACK, lw=0.9, label=nm)
    a.set_xticks(x); a.set_xticklabels([names[v] for v in s.index], fontsize=19); a.tick_params(axis="x", which="both", top=False, length=0)
    a.set_ylim(0, 1.32); a.set_ylabel("Hold-out R$^2$"); a.legend(loc="upper left", ncol=6, fontsize=16, columnspacing=0.9, handlelength=1.2)
    a.text(0.995, 0.97, l, transform=a.transAxes, ha="right", va="top", fontsize=25.7, fontweight="bold"); a.yaxis.set_minor_locator(AutoMinorLocator(2))
    a.set_title(ttl, fontsize=19.6, fontweight="bold")
fig.tight_layout(); fig.savefig("../figures/trt_step4b_ml.png"); plt.close(fig); print("ok")
