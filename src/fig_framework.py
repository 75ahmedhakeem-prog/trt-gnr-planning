from plotstyle import *
plt.rcParams.update({"font.size": 16.5*1.35, "axes.labelsize": 18*1.35, "legend.fontsize": 13.5*1.35, "xtick.major.size": 8, "ytick.major.size": 8})
_lab = lab
def lab(ax, s, x=0.04, y=0.95, ha="left", **k): ax.text(x, y, s, transform=ax.transAxes, ha=ha, va="top", fontsize=19*1.35, fontweight="bold", **k)
from matplotlib.patches import Rectangle, Circle, Ellipse, FancyArrowPatch, FancyBboxPatch
fig = plt.figure(figsize=(15, 6.6))
a = fig.add_axes([0.0, 0.02, 0.33, 0.9]); a.set_xlim(0, 10); a.set_ylim(0, 11); a.axis("off")
a.add_patch(Rectangle((0.3, 7.0), 9.4, 1.0, fc="#E9C9A0", ec=BLACK, lw=2)); a.text(9.5, 7.5, "Skin", ha="right", va="center", fontsize=21.6)
a.add_patch(Rectangle((0.3, 0.6), 9.4, 6.4, fc="#FFF4D6", ec=BLACK, lw=2)); a.text(9.5, 1.1, "Fat", ha="right", va="center", fontsize=21.6)
a.add_patch(Circle((5.0, 3.9), 1.9, fc="#F2A65A", ec=BLACK, lw=2)); a.text(5.0, 3.9, "Tumour\n+ GNRs", ha="center", va="center", fontsize=20.2)
for x in np.linspace(3.4, 6.6, 5): a.add_patch(FancyArrowPatch((x, 10.0), (x, 8.1), arrowstyle="-|>", mutation_scale=20, color=RED, lw=2.6))
a.text(5.6, 10.15, "NIR laser, 60 min", ha="center", va="bottom", fontsize=20.2, color=RED)
for k in range(3): a.add_patch(FancyArrowPatch((0.6, 5.0 - 0.8 * k), (2.9, 4.6 - 0.8 * k), arrowstyle="-|>", mutation_scale=16, color=BLUE, lw=2.2))
a.text(0.4, 6.3, "X-rays, 2 Gy", ha="left", va="center", fontsize=20.2, color=BLUE)
a.annotate("", xy=(3.1, 1.5), xytext=(6.9, 1.5), arrowprops=dict(arrowstyle="<->", lw=1.5)); a.text(5.0, 1.15, "D$_t$", fontsize=20.2, ha="center", va="top")
a.annotate("", xy=(8.0, 5.8), xytext=(8.0, 7.0), arrowprops=dict(arrowstyle="<->", lw=1.5)); a.text(8.2, 6.4, "d", fontsize=20.2)
a.text(0.2, 10.9, "(a)", fontsize=27.0, fontweight="bold", va="top")
b = fig.add_axes([0.35, 0.02, 0.64, 0.9]); b.set_xlim(0, 20); b.set_ylim(0, 11); b.axis("off")
boxes = [(1.0, 7.6, "Gans optics\nμ$_{a,GNR}$(λ)"), (5.6, 7.6, "δP1 light\ntransport"), (10.2, 7.6, "Pennes\nbioheat\nT(r, z, t)"), (14.8, 7.6, "Extended LQ\nα(T, t$_{int}$)\nβ(T, t$_{int}$)"),
         (14.8, 3.6, "EQD$_{RT}$, TER,\nTCP, TGF"), (10.2, 3.6, "Dataset\n288,000\nsamples"), (5.6, 3.6, "ANN\nsurrogate"), (1.0, 3.6, "Optimal\nprotocol\n(δP1 / MC)")]
for x, y, t in boxes:
    b.add_patch(FancyBboxPatch((x, y), 3.6, 2.4, boxstyle="round,pad=0.12", fc="white", ec=BLACK, lw=2)); b.text(x + 1.8, y + 1.2, t, ha="center", va="center", fontsize=18.2, linespacing=1.15)
for (x0, y0), (x1, y1) in [((4.75, 8.8), (5.45, 8.8)), ((9.35, 8.8), (10.05, 8.8)), ((13.95, 8.8), (14.65, 8.8)), ((16.6, 7.4), (16.6, 6.2)),
                           ((14.65, 4.8), (13.95, 4.8)), ((10.05, 4.8), (9.35, 4.8)), ((5.45, 4.8), (4.75, 4.8))]:
    b.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=18, lw=2.2, color=BLACK, shrinkA=0, shrinkB=0))
b.text(16.6, 1.6, "SiHa parameters\n(μ = 0.5 h$^{-1}$)", ha="center", va="center", fontsize=17.6)
b.text(2.8, 1.6, "T$_{skin}$ ≤ 43 °C, T$_{tum}$ ≤ 45 °C\nI ≤ MPE", ha="center", va="center", fontsize=17.6)
b.text(0.2, 10.9, "(b)", fontsize=27.0, fontweight="bold", va="top")
fig.savefig("../figures/trt_fig1_framework.png"); plt.close(fig); print("ok")
