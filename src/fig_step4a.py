from plotstyle import *
plt.rcParams.update({"font.size": 16.5*1.35, "axes.labelsize": 18*1.35, "legend.fontsize": 13.5*1.35, "xtick.major.size": 8, "ytick.major.size": 8})
_lab = lab
def lab(ax, s, x=0.04, y=0.95, ha="left", **k): ax.text(x, y, s, transform=ax.transAxes, ha=ha, va="top", fontsize=19*1.35, fontweight="bold", **k)
import pandas as pd
b = pd.read_csv("../data/trt_bestTER_gnr0.2.csv"); dep = sorted(b.depth_mm.unique()); dts = sorted(b.Dt_mm.unique())
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.9), sharey=True, gridspec_kw=dict(wspace=0.08))
for a, lam, l in ((axs[0], 808, "(a) 808 nm"), (axs[1], 1064, "(b) 1064 nm")):
    M_ = np.full((len(dep), len(dts)), 1.0)
    for _, r in b[b.lam_nm == lam].iterrows(): M_[dep.index(r.depth_mm), dts.index(r.Dt_mm)] = r.TER_D98
    im = a.imshow(M_, origin="upper", cmap="viridis", vmin=1.0, vmax=1.45, aspect="auto", extent=[-0.5, len(dts) - 0.5, len(dep) - 0.5, -0.5])
    for i in range(len(dep)):
        for j in range(len(dts)): a.text(j, i, f"{M_[i, j]:.2f}", ha="center", va="center", fontsize=16.9, color="white" if M_[i, j] < 1.3 else "black")
    a.set_xticks(range(len(dts))); a.set_xticklabels([f"{x:g}" for x in dts]); a.set_yticks(range(len(dep))); a.set_yticklabels([f"{x:g}" for x in dep])
    a.set_xlabel("Tumour diameter (mm)"); a.set_title(l + ", I ≤ MPE", fontsize=20.2, fontweight="bold"); a.tick_params(which="both", top=False, right=False)
axs[0].set_ylabel("Depth below skin (mm)")
cb = fig.colorbar(im, ax=axs, fraction=0.03, pad=0.02); cb.set_label("Best tumour TER (D$_{98}$)", fontsize=20.2)
fig.savefig("../figures/trt_step4a_maps.png"); plt.close(fig); print("ok")
