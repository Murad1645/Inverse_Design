import numpy as np, pandas as pd
from xgboost import XGBRegressor

# ---- load data + train surrogates (same as your pipeline) ----
df = pd.read_csv("canonical_dataset_v3_dfn.csv")
if "solver_ok" in df.columns:
    df = df[df["solver_ok"] == True]
FEAT = ["eps", "b", "Rp_um", "L_um"]
X = df[FEAT].values
def xgb(): return XGBRegressor(n_estimators=800, learning_rate=0.03, max_depth=4,
                               subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                               random_state=42, n_jobs=-1, tree_method="hist")
mQ = xgb().fit(X, df["Q_ratio"].values)
mD = xgb().fit(X, df["dV_3C"].values)

# ---- baseline ----
eps0, b0, Rp0, L0 = 0.35, 1.5, 7.0, 100.0
S0 = (1 - eps0) * L0          # = 65
print(f"baseline: S0 = {S0:.1f}, (eps=0.35, b=1.5, Rp=7, L=100)\n")

rng = np.random.default_rng(0)
def pareto_knee(C):
    q = mQ.predict(C); d = mD.predict(C)
    order = np.argsort(-q); front = []; best = np.inf
    for i in order:
        if d[i] < best - 1e-9:
            front.append(i); best = d[i]
    front = np.array(front)
    qn = (q[front] - q[front].min()) / np.ptp(q[front])
    dn = (d[front] - d[front].min()) / np.ptp(d[front])
    knee = front[np.argmax(qn - dn)]
    return C[knee], q[knee], d[knee]

# ---- (A) CONSTRAINED: loading-matched (your paper's method) ----
N = 200000
eps = rng.uniform(0.28, 0.45, N); b = rng.uniform(1.0, 2.5, N); Rp = rng.uniform(2, 12, N)
L = S0 / (1 - eps)                        # enforce S0 = 65
m = (L >= 50) & (L <= 140)
Cc = np.column_stack([eps[m], b[m], Rp[m], L[m]])
xc, qc, dc = pareto_knee(Cc)
print("CONSTRAINED (loading-matched) optimum:")
print(f"  eps={xc[0]:.3f}, b={xc[1]:.3f}, Rp={xc[2]:.2f} um, L={xc[3]:.1f} um")
print(f"  S0 = (1-eps)L = {(1-xc[0])*xc[3]:.1f},  Q_ratio = {qc:.3f}\n")

# ---- (B) UNCONSTRAINED: L sampled freely (no loading constraint) ----
eps = rng.uniform(0.28, 0.45, N); b = rng.uniform(1.0, 2.5, N)
Rp = rng.uniform(2, 12, N);       L = rng.uniform(50, 140, N)   # L is FREE now
Cu = np.column_stack([eps, b, Rp, L])
xu, qu, du = pareto_knee(Cu)
S0_u = (1 - xu[0]) * xu[3]
print("UNCONSTRAINED optimum:")
print(f"  eps={xu[0]:.3f}, b={xu[1]:.3f}, Rp={xu[2]:.2f} um, L={xu[3]:.1f} um")
print(f"  S0 = (1-eps)L = {S0_u:.1f},  Q_ratio = {qu:.3f}")
print(f"  loading reduction vs baseline: {100*(1 - S0_u/S0):.0f}% below S0=65")