"""Surrogates for the thermoradiotherapy dataset: ANN vs gradient-boosted trees (HistGB), random split and level hold-out."""
import json, numpy as np, pandas as pd, time
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
SEED = 42
FEAT = ["lam", "spot_mm", "log_h", "depth_mm", "Dt_mm", "perf", "log_gnr", "f_mpe", "tint_h"]
HOLD = {"spot_mm": [20], "depth_mm": [1, 2, 3, 4], "Dt_mm": [7.5, 10.0, 12.5], "perf": [1e-3, 2e-3], "mua_gnr": [0.1, 0.2], "f_mpe": [0.3, 0.5, 0.7, 0.9], "tint_h": [0.5, 1.0, 2.0]}
def prep(d):
    return pd.DataFrame({"lam": (d.lam_nm == 1064).astype(float), "spot_mm": d.spot_mm, "log_h": np.log10(d.h_cool), "depth_mm": d.depth_mm, "Dt_mm": d.Dt_mm,
                         "perf": d.perf * 1e3, "log_gnr": np.log10(d.mua_gnr), "f_mpe": d.f_mpe, "tint_h": d.tint_h})[FEAT]
def ann(seed=SEED):
    return TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(128, 128, 64), activation="tanh", alpha=1e-5,
        learning_rate_init=2e-3, batch_size=512, max_iter=300, early_stopping=True, validation_fraction=0.1, n_iter_no_change=25, random_state=seed)), transformer=StandardScaler())
def gbt(): return HistGradientBoostingRegressor(max_iter=600, max_leaf_nodes=127, learning_rate=0.05, random_state=SEED)
def metr(y, p): return dict(R2=float(r2_score(y, p)), RMSE=float(np.sqrt(mean_squared_error(y, p))))
if __name__ == "__main__":
    d = pd.read_csv("../data/trt_dataset_v2.csv.gz"); d = d[d.T_tum_max <= 45.0].reset_index(drop=True); X = prep(d); rng = np.random.default_rng(SEED); res = {"random": {}, "holdout": []}
    for tg in ("TER_D98", "T_skin_max"):
        y = d[tg].values; idx = rng.permutation(len(d)); n = int(0.8 * len(d)); tr, te = idx[:n][:60000], idx[n:][:20000]
        for nm, mk in (("ANN", ann), ("GBT", gbt)):
            t0 = time.time(); m = mk().fit(X.iloc[tr], y[tr]); res["random"].setdefault(tg, {})[nm] = {**metr(y[te], m.predict(X.iloc[te])), "t": round(time.time() - t0, 1)}
        print(tg, res["random"][tg], flush=True)
        for var, lvs in HOLD.items():
            for lv in lvs:
                mask = np.isclose(d[var].values, lv); trp = rng.choice(np.where(~mask)[0], min(60000, (~mask).sum()), replace=False); tep = rng.choice(np.where(mask)[0], min(20000, mask.sum()), replace=False)
                row = dict(target=tg, variable=var, level=lv)
                for nm, mk in (("ANN", ann), ("GBT", gbt)):
                    m = mk().fit(X.iloc[trp], y[trp]); mm = metr(y[tep], m.predict(X.iloc[tep])); row[f"{nm}_R2"] = mm["R2"]; row[f"{nm}_RMSE"] = mm["RMSE"]
                res["holdout"].append(row); print(row, flush=True)
                json.dump(res, open("../data/ml_trt_v2.json", "w"), indent=1)
    pd.DataFrame(res["holdout"]).to_csv("../data/ml_trt_holdout_v2.csv", index=False); json.dump(res, open("../data/ml_trt_v2.json", "w"), indent=1); print("done")
