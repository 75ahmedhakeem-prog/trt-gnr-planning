"""Verification of the axisymmetric Pennes solver against the 1-D steady analytical solution (uniform slab, exponential source)."""
import os
import sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import design_sim as ds
k0, rhoc0, perf = 0.5, 4.0e6, 1e-2; w = perf * ds.RHOC_B; mu = 300.0; Q0 = 1e5; h = 200.0; H = ds.H * 1e-3
k = np.full((ds.TR, ds.TZ), k0); rhoc = np.full((ds.TR, ds.TZ), rhoc0); wrc = np.full((ds.TR, ds.TZ), w)
z = ds.zt * 1e-3; Q = Q0 * np.exp(-mu * z)[None, :] * np.ones((ds.TR, 1))
F = ds.heat_fields(Q, k, rhoc, wrc, t_end=7200.0, every=600.0, h_top=h)
num = F[-1][0]; num2 = F[-2][0]
m = np.sqrt(w / k0); A = Q0 / (w - k0 * mu**2)
# T = A e^{-mu z} + C1 e^{m z} + C2 e^{-m z};  k T'(0) = h T(0);  T(H) = 0
M = np.array([[k0 * m - h, -k0 * m - h], [np.exp(m * H), np.exp(-m * H)]]); rhs = np.array([h * A + k0 * mu * A, -A * np.exp(-mu * H)])
C1, C2 = np.linalg.solve(M, rhs); ana = A * np.exp(-mu * z) + C1 * np.exp(m * z) + C2 * np.exp(-m * z)
err = np.abs(num - ana); sel = ds.zt <= 15
res = dict(max_rise_analytic=float(ana.max()), max_rise_numeric=float(num.max()), max_abs_error=float(err[sel].max()), rel_error_at_peak=float(abs(num.max() - ana.max()) / ana.max()), steady_change_last_600s=float(np.abs(num - num2).max()))
print(res); json.dump(res, open("../data/verify_heat.json", "w"), indent=1)
