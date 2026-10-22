from plotstyle import *
plt.rcParams.update({"font.size": 16.5*1.35, "axes.labelsize": 18*1.35, "legend.fontsize": 13.5*1.35, "xtick.major.size": 8, "ytick.major.size": 8})
_lab = lab
def lab(ax, s, x=0.04, y=0.95, ha="left", **k): ax.text(x, y, s, transform=ax.transAxes, ha=ha, va="top", fontsize=19*1.35, fontweight="bold", **k)
import json
exec(open("step2.py").read().split("res = {}")[0])
fig, axs = plt.subplots(1, 3, figsize=(16, 5.6), gridspec_kw=dict(wspace=0.5))
a = axs[0]; tt = np.concatenate([np.linspace(0, 2, 41), np.linspace(2.1, 30, 140)])
for A, col, ls in ((0.0, BLUE, "-"), (0.2, GREEN, "--"), (0.45, RED, "-.")):
    a.plot(tt, [d98(eqd_fx(t, A)) for t in tt], ls, color=col, lw=2.4, label=f"A = {A:.2f}")
a.vlines([0.5, 24], 2.2, 3.4, color=GREY, ls=":", lw=1.3)
a.set_xlabel("Interval (h)"); a.set_ylabel("EQD, D$_{98}$ (Gy)"); a.set_xlim(0, 30); a.set_ylim(2.2, 4.2)
a.legend(loc="upper right", fontsize=17.6); a.xaxis.set_minor_locator(AutoMinorLocator(2)); a.yaxis.set_minor_locator(AutoMinorLocator(2)); lab(a, "(a)", x=0.06, y=0.93)
b = axs[1]; Ts = np.linspace(39, 43.5, 19)
for mu, col, mk in ((0.3, GREEN, "s"), (0.5, RED, "o"), (1.0, BLUE, "^")):
    pp = dict(C.TUM, mu=mu); out = []
    for T in Ts:
        TT = np.full(Tm.shape, T); KDs = k_heat(T, C.TUM) * 3600.0
        def e(t, A):
            aa, bb = lq(TT, t, pp); a37 = pp["a37"]; b37 = a37 / pp["ab37"]; den = a37 + b37 * 2
            return float(((aa * (1 + A * g(t)) * 2 + bb * 4) / den + KDs / den)[0, 0])
        out.append(brentq(lambda A: e(24.0, A) - e(0.5, 0), 0, 10))
    b.plot(Ts, out, "-" + mk, color=col, ms=7, lw=2.2, label=f"μ = {mu} h$^{{-1}}$")
b.set_xlabel("Tumour T (°C)"); b.set_ylabel("Break-even gain A*"); b.set_ylim(0, 1.3)
b.legend(loc="upper left", fontsize=17.6); b.xaxis.set_minor_locator(AutoMinorLocator(2)); b.yaxis.set_minor_locator(AutoMinorLocator(2)); lab(b, "(b)", x=0.96, ha="right", y=0.2)
c = axs[2]; f1 = np.linspace(0.15, 0.7, 120); r = np.linspace(1, 4, 120); F1, Rr = np.meshgrid(f1, r)
Ag = ((F1 * Rr + (1 - F1)) / (0.15 * Rr + 0.85)) - 1
im = c.contourf(F1 * 100, Rr, Ag, levels=np.linspace(0, 1.2, 13), cmap="viridis")
cs = c.contour(F1 * 100, Rr, Ag, levels=[0.45], colors="white", linewidths=2.5); c.clabel(cs, fmt="A* = 0.45", fontsize=17.6)
cb = fig.colorbar(im, ax=c, fraction=0.05, pad=0.03); cb.set_label("Gain A", fontsize=20.2)
c.set_xlabel("G2/M at 24 h (%)"); c.set_ylabel("G2/M sensitivity r")
c.text(0.04, 0.96, "(c)", transform=c.transAxes, ha="left", va="top", fontsize=25.7, fontweight="bold", color="white")
fig.savefig("../figures/trt_step2.png"); plt.close(fig); print("ok")
