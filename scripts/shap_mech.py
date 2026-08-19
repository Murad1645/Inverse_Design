import numpy as np, pandas as pd, shap
from xgboost import XGBRegressor

df = pd.read_csv("canonical_dataset_v3_dfn.csv")
if "solver_ok" in df: df = df[df["solver_ok"]==True]
FEAT = ["eps","b","Rp_um","L_um"]; X = df[FEAT].copy()

def xgb(): return XGBRegressor(n_estimators=800, learning_rate=0.03, max_depth=4,
    subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, random_state=42, n_jobs=-1, tree_method="hist")

for tname in ["Q_ratio","dV_3C"]:
    y = df[tname].values
    m = xgb(); m.fit(X, y)
    expl = shap.TreeExplainer(m); sv = expl.shap_values(X)
    mean_abs = np.abs(sv).mean(0)
    order = np.argsort(mean_abs)[::-1]
    print(f"=== {tname}: global SHAP importance (mean|SHAP|) ===")
    for i in order:
        # directional sign: corr between feature value and its shap value
        dirn = np.corrcoef(X[FEAT[i]].values, sv[:,i])[0,1]
        arrow = "raises" if dirn>0 else "lowers"
        print(f"   {FEAT[i]:6s}  {mean_abs[i]:.4f}   (higher {FEAT[i]} {arrow} {tname})")
    # interaction check for Rp: does Rp's SHAP depend strongly on L or b?
    rp_i = FEAT.index("Rp_um")
    inter = expl.shap_interaction_values(X.iloc[:400])  # subset for speed
    rp_self = np.abs(inter[:, rp_i, rp_i]).mean()
    rp_L = np.abs(inter[:, rp_i, FEAT.index("L_um")]).mean()*2
    rp_b = np.abs(inter[:, rp_i, FEAT.index("b")]).mean()*2
    print(f"   [Rp interaction] main={rp_self:.4f}  Rp×L={rp_L:.4f}  Rp×b={rp_b:.4f}")
    print()
