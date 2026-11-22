"""Smooth-function baselines for the surrogate comparison (reviewer comment): polynomial response surfaces, additive +
pairwise-interaction cubic splines, local multiquadric RBF interpolation and Gaussian-process regression (ARD Matern 5/2).
The random split and the 38 level hold-outs of ml_trt.py are replayed exactly (same generator, same call sequence), so
every baseline is trained and tested on the same samples as the ANN and GBT of ml_trt_holdout_v2.csv."""
import os, sys, json, time; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, SplineTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.interpolate import RBFInterpolator
import ml_trt as M

N_GP_OPT, N_GP = 1500, 6000          # GP: hyperparameters optimised on 1500 training points, posterior on 6000

def poly(deg): return make_pipeline(StandardScaler(), PolynomialFeatures(deg), Ridge(alpha=1e-6, solver="cholesky"))
def spline():
    cont = [i for i, f in enumerate(M.FEAT) if f != "lam"]
    ct = ColumnTransformer([("s", SplineTransformer(n_knots=5, degree=3, knots="quantile", extrapolation="linear"), cont)], remainder="passthrough")
    return make_pipeline(ct, PolynomialFeatures(2, interaction_only=True, include_bias=False), StandardScaler(), Ridge(alpha=1e-3, solver="cholesky"))
class RBF:
    def fit(self, X, y):
        self.s = StandardScaler().fit(X); self.m = y.mean()
        self.f = RBFInterpolator(self.s.transform(X), y - self.m, kernel="multiquadric", epsilon=1.0, neighbors=64, smoothing=1e-6, degree=0); return self
    def predict(self, X): return self.f(self.s.transform(X)) + self.m
class GP:
    def __init__(self, seed=0): self.seed = seed
    def fit(self, X, y):
        rng = np.random.default_rng(self.seed); X = np.asarray(X, float)
        self.sx = StandardScaler().fit(X); self.my, self.sy = y.mean(), y.std(); Z = self.sx.transform(X); t = (y - self.my) / self.sy
        k = ConstantKernel(1.0) * Matern(length_scale=np.ones(X.shape[1]), length_scale_bounds=(1e-2, 1e3), nu=2.5) + WhiteKernel(1e-4, (1e-8, 1e-1))
        i = rng.choice(len(Z), min(N_GP_OPT, len(Z)), replace=False)
        g = GaussianProcessRegressor(k, normalize_y=False, n_restarts_optimizer=0, random_state=self.seed).fit(Z[i], t[i])
        j = rng.choice(len(Z), min(N_GP, len(Z)), replace=False)
        self.g = GaussianProcessRegressor(g.kernel_, optimizer=None).fit(Z[j], t[j]); return self
    def predict(self, X):
        Z = self.sx.transform(np.asarray(X, float))
        return np.concatenate([self.g.predict(Z[i:i + 1000]) for i in range(0, len(Z), 1000)]) * self.sy + self.my
MODELS = {"Poly2": lambda: poly(2), "Poly3": lambda: poly(3), "Poly4": lambda: poly(4), "Poly5": lambda: poly(5), "Spline": spline, "RBF": lambda: RBF(), "GP": lambda: GP()}

def fit_eval(nm, Xtr, ytr, Xte, yte):
    t0 = time.time(); m = MODELS[nm]().fit(Xtr, ytr); t1 = time.time(); p = m.predict(Xte); t2 = time.time()
    return dict(M.metr(yte, p), t_fit=t1 - t0, t_pred_us=(t2 - t1) / len(Xte) * 1e6)

if __name__ == "__main__":
    only = sys.argv[1].split(",") if len(sys.argv) > 1 else list(MODELS)
    d = pd.read_csv("../data/trt_dataset_v2.csv.gz"); d = d[d.T_tum_max <= 45.0].reset_index(drop=True); X = M.prep(d).values
    rng = np.random.default_rng(M.SEED); res = {"random": {}, "holdout": []}
    for tg in ("TER_D98", "T_skin_max"):
        y = d[tg].values; idx = rng.permutation(len(d)); n = int(0.8 * len(d)); tr, te = idx[:n][:60000], idx[n:][:20000]
        for nm in only:
            res["random"].setdefault(tg, {})[nm] = fit_eval(nm, X[tr], y[tr], X[te], y[te])
        print(tg, "random", {k: round(v["R2"], 4) for k, v in res["random"][tg].items()}, flush=True)
        for var, lvs in M.HOLD.items():
            for lv in lvs:
                mask = np.isclose(d[var].values, lv); trp = rng.choice(np.where(~mask)[0], min(60000, (~mask).sum()), replace=False)
                tep = rng.choice(np.where(mask)[0], min(20000, mask.sum()), replace=False); row = dict(target=tg, variable=var, level=lv)
                for nm in only:
                    r = fit_eval(nm, X[trp], y[trp], X[tep], y[tep]); row[f"{nm}_R2"] = r["R2"]; row[f"{nm}_RMSE"] = r["RMSE"]; row[f"{nm}_tfit"] = r["t_fit"]
                res["holdout"].append(row); print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if not k.endswith("tfit")}, flush=True)
    tag = "_".join(only) if len(sys.argv) > 1 else "all"
    pd.DataFrame(res["holdout"]).to_csv(f"../data/ml_baselines_holdout_{tag}.csv", index=False)
    json.dump(res, open(f"../data/ml_baselines_{tag}.json", "w"), indent=1)
