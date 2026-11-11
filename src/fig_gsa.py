from plotstyle import *
import json
d = json.load(open("../data/gsa.json")); df = pd.read_csv("../data/gsa_samples.csv.gz")
NAMES = {"ra": "α$_{41}$/α$_{37}$", "rb": "β$_{41}$/β$_{37}$", "mu": "μ", "ab": "α/β$_{tum}$", "a37": "α$_{37}$", "direct": "Direct kill",
         "ab_n": "α/β$_{nt}$", "mu_n": "μ$_{nt}$", "perf": "ω$_b$", "gnr": "μ$_{a,GNR}$", "h": "h", "fs": "μ$_s$′ scale", "fa": "μ$_a$ scale", "rho": "ρ$_{cl}$"}
ref = d["reference"]["dD50"]
def binned(x, y, edges):
    c, m = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (x >= lo) & (x < hi)
        if s.sum() > 20: c.append(np.median(x[s])); m.append(np.median(y[s]))
    return np.array(c), np.array(m)
fig, axs = plt.subplots(2, 2, figsize=(13.5, 10.5), gridspec_kw=dict(wspace=0.32, hspace=0.30))
c_ax, d_ax, a_ax, b_ax = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]
# (a) dose shift versus alpha ratio for three ranges of mu
for (lo, hi), col, mk, lb in (((0.25, 0.5), RED, "o", "μ < 0.5 h$^{-1}$"), ((0.5, 1.0), GREEN, "s", "0.5–1 h$^{-1}$"), ((1.0, 2.01), BLUE, "^", "μ > 1 h$^{-1}$")):
    s = (df.mu >= lo) & (df.mu < hi); x, y = binned(df.ra[s].values, df.dD50[s].values, np.linspace(1.2, 2.6, 9))
    a_ax.plot(x, y, "-" + mk, color=col, ms=8, label=lb)
a_ax.plot([1.73], [ref], "*", color=BLACK, ms=17, label="Reference case")
a_ax.set_xlabel("α$_{41}$/α$_{37}$"); a_ax.set_ylabel("ΔD$_{50}$ (Gy)"); a_ax.set_xlim(1.15, 2.65); a_ax.set_ylim(0, 20)
a_ax.legend(loc="upper left", bbox_to_anchor=(0.12, 1.0), fontsize=13.5); lab(a_ax, "(a)", x=0.04, y=0.95)
# (b) in vivo decay of sensitisation (Overgaard 1980) with fitted curves
bo = json.load(open("../data/bench_overgaard.json"))["fits"]; tt = np.linspace(0, 25, 300)
for k, col, mk, lb in (("tumour, heat before RT", RED, "o", "Tumour, heat → RT"), ("skin, RT before heat", BLUE, "s", "Skin, RT → heat")):
    q = bo[k]; b_ax.plot(q["t"], q["TER"], mk, color=col, ms=9, mfc="none", mew=2, label=lb + f" (μ = {q['mu']:.2f} h$^{{-1}}$)")
    b_ax.plot(tt, np.exp(np.log(q["TER_inf"]) + q["K"] * np.exp(-q["mu"] * tt)), "-", color=col, lw=2.0)
b_ax.set_xlabel("Interval (h)"); b_ax.set_ylabel("TER (in vivo)"); b_ax.set_xlim(-0.8, 25); b_ax.set_ylim(0.6, 3.4)
b_ax.legend(loc="upper right", fontsize=13); lab(b_ax, "(b)", x=0.04, y=0.95)
b_ax.xaxis.set_minor_locator(AutoMinorLocator(2))
# (c) distribution of the dose shift
v = df.dD50.values; p = np.percentile(v, [2.5, 50, 97.5])
c_ax.hist(v, bins=np.arange(0, 40, 1.0), histtype="step", color=BLUE, lw=2.4)
for q, ls in zip(p, ("--", "-", "--")): c_ax.axvline(q, color=BLACK, ls=ls, lw=1.6)
c_ax.axvline(ref, color=RED, ls=":", lw=2.2)
c_ax.set_xlabel("ΔD$_{50}$ (Gy)"); c_ax.set_ylabel("Samples"); c_ax.set_xlim(0, 40); c_ax.set_ylim(0, 520)
c_ax.text(0.97, 0.95, f"Median {p[1]:.1f} Gy\n95%: {p[0]:.1f}–{p[2]:.1f} Gy\nReference {ref:.1f} Gy", transform=c_ax.transAxes, ha="right", va="top", fontsize=14.5)
lab(c_ax, "(c)", x=0.06, y=0.95)
# (d) first-order sensitivity indices
S = d["S1"]; keys = sorted(S["dD50"], key=lambda k: -S["dD50"][k])[:8][::-1]; yy = np.arange(len(keys))
d_ax.barh(yy + 0.2, [S["dD50"][k] for k in keys], 0.38, color=BLUE, label="ΔD$_{50}$")
d_ax.barh(yy - 0.2, [S["TER"][k] for k in keys], 0.38, color=ORANGE, label="TER")
d_ax.set_yticks(yy); d_ax.set_yticklabels([NAMES[k] for k in keys], fontsize=14); d_ax.tick_params(axis="y", which="both", right=False)
d_ax.set_xlabel("First-order index S$_1$"); d_ax.set_xlim(0, 0.5)
d_ax.legend(loc="lower right", fontsize=13.5); lab(d_ax, "(d)", x=0.96, y=0.95, ha="right")
for a in (a_ax, c_ax): a.xaxis.set_minor_locator(AutoMinorLocator(2)); a.yaxis.set_minor_locator(AutoMinorLocator(2))
b_ax.yaxis.set_minor_locator(AutoMinorLocator(2)); d_ax.xaxis.set_minor_locator(AutoMinorLocator(2))
fig.savefig("../figures/trt_gsa.png"); plt.close(fig); print("ok", p)
