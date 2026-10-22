from plotstyle import *
plt.rcParams.update({"font.size": 16.5*1.35, "axes.labelsize": 18*1.35, "legend.fontsize": 13.5*1.35, "xtick.major.size": 8, "ytick.major.size": 8})
_lab = lab
def lab(ax, s, x=0.04, y=0.95, ha="left", **k): ax.text(x, y, s, transform=ax.transAxes, ha=ha, va="top", fontsize=19*1.35, fontweight="bold", **k)
import json
r = json.load(open("../data/step3.json"))
fig, axs = plt.subplots(1, 2, figsize=(13, 5.0), gridspec_kw=dict(wspace=0.3))
a = axs[0]; cols = {"100 kVp (2 mm Al)": RED, "250 kVp (1 mm Cu)": ORANGE, "6 MV + 10% low-energy scatter": BLUE}
names = {"100 kVp (2 mm Al)": "100 kVp", "250 kVp (1 mm Cu)": "250 kVp", "6 MV + 10% low-energy scatter": "6 MV"}
for b, col in cols.items():
    c = np.array([float(k) for k in r[b]]); th = [v["TER_heat_plus_DEF"] for v in r[b].values()]; dn = [v["TER_DEF_only"] for v in r[b].values()]
    a.plot(c, th, "-o", color=col, ms=6, lw=2.3, label=f"{names[b]}: heat + gold"); a.plot(c, dn, "--", color=col, lw=2.0, label=f"{names[b]}: gold only")
a.set_xscale("log"); a.set_xlabel("Gold in tumour (μg g$^{-1}$)"); a.set_ylabel("Tumour TER (D$_{98}$)"); a.set_ylim(0.95, 2.8)
a.legend(loc="upper left", fontsize=15.5, ncol=1, bbox_to_anchor=(0.0, 0.88)); a.yaxis.set_minor_locator(AutoMinorLocator(2)); lab(a, "(a)")
b = axs[1]; v = r["100 kVp (2 mm Al)"]; c = np.array([float(k) for k in v])
b.plot(c, [x["T50"] for x in v.values()], "-o", color=RED, ms=7, lw=2.3, label="Tumour T$_{50}$"); b.plot(c, [x["T90"] for x in v.values()], "-s", color=ORANGE, ms=7, lw=2.3, label="Tumour T$_{90}$")
b.plot(c, [x["Tskin"] for x in v.values()], "-^", color=BLUE, ms=7, lw=2.3, label="Skin T$_{max}$")
b.set_xscale("log"); b.set_xlabel("Gold in tumour (μg g$^{-1}$)"); b.set_ylabel("Temperature (°C)"); b.set_ylim(37.5, 44)
b.legend(loc="lower left", fontsize=17.6); b.yaxis.set_minor_locator(AutoMinorLocator(2)); lab(b, "(b)")
fig.savefig("../figures/trt_step3.png"); plt.close(fig); print("ok")
