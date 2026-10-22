"""Gold mass concentration needed for a given nanorod absorption coefficient (Gans theory)."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, gans
lam = np.arange(600, 1300.1, 1.0)
def mua_per_ugml(L, Dm):
    V = np.pi / 6 * L * Dm**2; N = 1.0 * 1e-3 / (V * 1e-21 * 19.3) / 1e3      # per 1 ug/mL 
    ar = L / Dm; ecc = np.sqrt(1 - 1 / ar**2); S = 2 * np.pi * (Dm / 2)**2 * (1 + (L / 2) / ((Dm / 2) * ecc) * np.arcsin(ecc))
    lo = gans.cabs(lam, L, Dm, eps=gans.eps_au(lam, width_nm=2 * 4 * V / S))[0]; hi = gans.cabs(lam, L, Dm, eps=gans.eps_au(lam))[0]
    return lam[np.argmax(hi)], N * lo.max() * 1e-15, N * hi.max() * 1e-15
out = {}
for L in np.arange(35, 90, 1.0):
    pk, lo, hi = mua_per_ugml(L, 10.0); out[L] = (pk, lo, hi)
for target in (808, 1064):
    L = min(out, key=lambda k: abs(out[k][0] - target)); pk, lo, hi = out[L]
    print(f"LSPR {target}: rod 10 x {L:.0f} nm (AR {L/10:.1f}), peak {pk:.0f} nm, mua per ug/mL = {lo:.4f}-{hi:.4f} mm^-1 -> mua 0.15 needs {0.15/hi:.1f}-{0.15/lo:.1f} ug/mL")
