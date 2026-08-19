import numpy as np, pandas as pd
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

df = pd.read_csv("canonical_dataset_v3_dfn.csv")
if "solver_ok" in df: df = df[df["solver_ok"]==True]
FEAT = ["eps","b","Rp_um","L_um"]; X = df[FEAT].values
targets = {"Q_ratio": df["Q_ratio"].values, "dV_3C": df["dV_3C"].values}

def xgb(): return XGBRegressor(n_estimators=800, learning_rate=0.03, max_depth=4,
    subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, random_state=42, n_jobs=-1, tree_method="hist")
def rf():  return RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
print(f"N = {len(df)} | 5-fold CV (shuffled, seed 42) | no test-set filtering\n")
for tname, y in targets.items():
    print(f"=== {tname} ===")
    for mname, mk in [("XGBoost", xgb), ("RandomForest", rf)]:
        r2s, rmses = [], []
        for tr, te in kf.split(X):
            m = mk(); m.fit(X[tr], y[tr]); p = m.predict(X[te])
            r2s.append(r2_score(y[te], p)); rmses.append(np.sqrt(mean_squared_error(y[te], p)))
        r2s, rmses = np.array(r2s), np.array(rmses)
        print(f"  {mname:12s} R2 = {r2s.mean():.3f} +/- {r2s.std():.3f}   RMSE = {rmses.mean():.4f} +/- {rmses.std():.4f}")
    print()
