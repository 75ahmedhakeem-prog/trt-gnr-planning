from plotstyle import *
plt.rcParams.update({"font.size": 16.5*1.35, "axes.labelsize": 18*1.35, "legend.fontsize": 13.5*1.35, "xtick.major.size": 8, "ytick.major.size": 8})
_lab = lab
def lab(ax, s, x=0.04, y=0.95, ha="left", **k): ax.text(x, y, s, transform=ax.transAxes, ha=ha, va="top", fontsize=19*1.35, fontweight="bold", **k)
import json, design_sim as ds, mc_het
# (a) heat solver vs analytical solution
k0, rhoc0, perf = 0.5, 4.0e6, 1e-2; w = perf * ds.RHOC_B; mu = 300.0; Q0 = 1e5; h = 200.0; H = ds.H * 1e-3
k = np.full((ds.TR, ds.TZ), k0); rhoc = np.full((ds.TR, ds.TZ), rhoc0); wrc = np.full((ds.TR, ds.TZ), w); z = ds.zt * 1e-3
Q = Q0 * np.exp(-mu * z)[None, :] * np.ones((ds.TR, 1)); num = ds.heat_fields(Q, k, rhoc, wrc, t_end=7200.0, every=600.0, h_top=h)[-1][0]
m = np.sqrt(w / k0); A = Q0 / (w - k0 * mu**2); M = np.array([[k0 * m - h, -k0 * m - h], [np.exp(m * H), np.exp(-m * H)]])
C1, C2 = np.linalg.solve(M, np.array([h * A + k0 * mu * A, -A * np.exp(-mu * H)])); zz = np.linspace(0, 20, 400) * 1e-3
ana = A * np.exp(-mu * zz) + C1 * np.exp(m * zz) + C2 * np.exp(-m * zz)
# (b) delta-P1 vs MC, on-axis absorbed power density (808 nm, reference tumour)
import os, pickle
CACHE = "../data/verify_dp1_mc_profiles.pkl"
res = pickle.load(open(CACHE, "rb")) if os.path.exists(CACHE) else {}
for lam in ([] if res else (808, 1064)):
    mua, mus, g, tum = ds.optics(lam, 2.5, 9.0, 0.15); P = 0.01 * np.pi * 12.5**2
    o = __import__("dp1").solve(ds.R, ds.H, ds.NR, ds.NZ, mua, mus, g, 12.5, P=P, n_in=1.4)
    mc = mc_het.run(ds.R, ds.H, ds.NR, ds.NZ, mua, mus, g, 12.5, P=P, n_in=1.4, n_top=1.0, n_side=1.4, n_bot=1.4, N=200000, seed=5)
    vol = 2 * np.pi * ds.rc[:, None] * ds.DR * ds.DZ
    res[lam] = dict(dP1=o["Q"][:8].mean(0), MC=mc["Q"][:8].mean(0), tum_dP1=float((o["Q"] * vol)[tum].sum()), tum_MC=float((mc["Q"] * vol)[tum].sum()),
                    skin_dP1=float((o["Q"] * vol)[:, ds.zc < 1.5].sum()), skin_MC=float((mc["Q"] * vol)[:, ds.zc < 1.5].sum()))
pickle.dump(res, open(CACHE, "wb"))
out = {lam: {k: v for k, v in r.items() if not hasattr(v, "shape")} for lam, r in res.items()}; print(out)
json.dump(out, open("../data/verify_dp1_mc.json", "w"), indent=1)
fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw=dict(wspace=0.3))
a = axs[0]; a.plot(zz * 1e3, ana, "-", color=BLACK, lw=2.4, label="Analytical"); a.plot(ds.zt, num, "o", color=RED, ms=6, mfc="white", mew=1.6, label="Finite volumes")
a.set_xlim(0, 20); a.set_xlabel("Depth (mm)"); a.set_ylabel("Temperature rise (°C)"); a.legend(loc="upper right"); lab(a, "(a)", y=0.2)
a.xaxis.set_minor_locator(AutoMinorLocator(2)); a.yaxis.set_minor_locator(AutoMinorLocator(2))
b = axs[1]
for lam, col in ((808, BLUE), (1064, RED)):
    b.semilogy(ds.zc, res[lam]["MC"] * 1e3, "-", color=col, lw=2.2, label=f"{lam} nm, MC"); b.semilogy(ds.zc, res[lam]["dP1"] * 1e3, "--", color=col, lw=2.2, label=f"{lam} nm, δP1")
b.axvline(1.5, color=GREY, ls=":", lw=1.3); b.axvline(4.0, color=GREY, ls="--", lw=1.3); b.axvline(13.0, color=GREY, ls="--", lw=1.3)
b.text(8.5, 4e-3, "tumour", ha="center", fontsize=18.9); b.set_xlim(0, 20); b.set_ylim(2e-3, 6)
b.set_xlabel("Depth (mm)"); b.set_ylabel("Absorbed power (mW mm$^{-3}$)"); b.legend(loc="upper right", fontsize=16.9); lab(b, "(b)", x=0.80, y=0.55)
fig.savefig("../figures/trt_fig2_verif.png"); plt.close(fig)
