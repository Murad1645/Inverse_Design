# =====================================================================
# MASTER NOTEBOOK — FIGURES
# Generates the four core paper figures. Run AFTER the surrogate models
# (mQ, mD) and the datasets are available. Colab paths assumed (/content).
# Figures follow the house style: titles and axis labels NOT bold.
# =====================================================================
import numpy as np, pandas as pd, os
import matplotlib.pyplot as plt
import shap
from xgboost import XGBRegressor

# ----- global style: non-bold labels/titles -----
plt.rcParams.update({
    "axes.titleweight": "normal", "axes.labelweight": "normal",
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
})

V3 = "canonical_dataset_v3_dfn.csv"
V2 = "canonical_dataset_v2.csv"          # SPMe (optional; for Fig 3)
FEAT = ["eps", "b", "Rp_um", "L_um"]
LAB  = {"eps": "ε (porosity)", "b": "b (Bruggeman)", "Rp_um": "Rₚ (µm)", "L_um": "L (µm)"}

df = pd.read_csv(V3)
if "solver_ok" in df: df = df[df["solver_ok"] == True]
X = df[FEAT].copy()

def xgb():
    return XGBRegressor(n_estimators=800, learning_rate=0.03, max_depth=4, subsample=0.9,
                        colsample_bytree=0.9, reg_lambda=1.0, random_state=42,
                        n_jobs=-1, tree_method="hist")

mQ = xgb().fit(X, df["Q_ratio"].values)
mD = xgb().fit(X, df["dV_3C"].values)

# ---------------------------------------------------------------------
# FIGURE 1 — SHAP feature importance (mechanism)
# ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, model, title in [(axes[0], mQ, "Rate capability (Q₃C/Q₀.₅C)"),
                         (axes[1], mD, "Polarization ΔV₃C (V)")]:
    sv = shap.TreeExplainer(model).shap_values(X)
    imp = np.abs(sv).mean(0); order = np.argsort(imp)
    ax.barh([LAB[FEAT[i]] for i in order], imp[order],
            color="#3b7dd8", edgecolor="black", linewidth=0.6)
    ax.set_xlabel("mean |SHAP value|"); ax.set_title(title)
plt.tight_layout(); plt.savefig("fig1_shap_importance.png", dpi=200, bbox_inches="tight")
plt.show()

# ---------------------------------------------------------------------
# FIGURE 2 — Pareto front (loading-matched inverse design)
# ---------------------------------------------------------------------
S0 = (1 - 0.35) * 100.0                          # baseline loading proxy = 65
rng = np.random.default_rng(0); N = 200000
eps = rng.uniform(0.28, 0.45, N); b = rng.uniform(1.0, 2.5, N); Rp = rng.uniform(2.0, 12.0, N)
L = S0 / (1 - eps); m = (L >= 50) & (L <= 140)
C = np.column_stack([eps[m], b[m], Rp[m], L[m]])
qp = mQ.predict(C); dp = mD.predict(C)

def pareto(q, d):
    o = np.argsort(-q); front = []; best = np.inf
    for i in o:
        if d[i] < best - 1e-9: front.append(i); best = d[i]
    return np.array(front)

pf = pareto(qp, dp)
qn = (qp[pf] - qp[pf].min()) / np.ptp(qp[pf]); dn = (dp[pf] - dp[pf].min()) / np.ptp(dp[pf])
knee = pf[np.argmax(qn - dn)]

fig, ax = plt.subplots(figsize=(6.2, 4.6))
ax.scatter(dp[::50], qp[::50], s=6, color="#cccccc", alpha=0.5, label="Loading-matched candidates")
order = pf[np.argsort(dp[pf])]
ax.plot(dp[order], qp[order], "-o", color="#3b7dd8", ms=4, lw=1.5, label="Pareto front")
ax.scatter(dp[knee], qp[knee], s=140, marker="*", color="#e4572e",
           edgecolor="black", zorder=5, label="Selected design")
ax.set_xlabel("Polarization ΔV₃C (V)  → lower better")
ax.set_ylabel("Rate capability Q₃C/Q₀.₅C  → higher better")
ax.set_title("Loading-matched Pareto optimization (surrogate)")
ax.legend(frameon=False, fontsize=9)
plt.tight_layout(); plt.savefig("fig2_pareto_front.png", dpi=200, bbox_inches="tight")
plt.show()

# ---------------------------------------------------------------------
# FIGURE 3 — Model-fidelity shift: SPMe vs DFN (needs both CSVs)
# ---------------------------------------------------------------------
if os.path.exists(V2):
    v2 = pd.read_csv(V2)
    if "solver_ok" in v2: v2 = v2[v2["solver_ok"] == True]
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    bins = np.linspace(0, 1, 26)
    ax.hist(v2["Q_ratio"], bins=bins, alpha=0.55, color="#e4572e",
            label=f"SPMe (v2): {100*(v2.Q_ratio<=0.2).mean():.0f}% fail at 3C")
    ax.hist(df["Q_ratio"], bins=bins, alpha=0.55, color="#3b7dd8",
            label=f"DFN (v3): {100*(df.Q_ratio<=0.2).mean():.0f}% fail at 3C")
    ax.axvline(0.2, ls="--", color="black", lw=1)
    ax.text(0.205, ax.get_ylim()[1]*0.9, "failure threshold (0.2)", fontsize=8)
    ax.set_xlabel("Rate capability Q₃C/Q₀.₅C")
    ax.set_ylabel("Number of designs")
    ax.set_title("Reduced-order (SPMe) vs full physics (DFN)")
    ax.legend(frameon=False, fontsize=9)
    plt.tight_layout(); plt.savefig("fig3_spme_vs_dfn.png", dpi=200, bbox_inches="tight")
    plt.show()
else:
    print("Fig 3 skipped: upload canonical_dataset_v2.csv to /content to generate the SPMe-vs-DFN figure.")

# ---------------------------------------------------------------------
# FIGURE 4 — Rate-capability curve (baseline vs optimized, DFN 1C–5C)
# ---------------------------------------------------------------------
# Values from the DFN 1C-5C robustness sweep (recompute with robustness.py if params change)
rates = [0.5, 1, 2, 3, 4, 5]
base  = [1.000, 0.978, 0.875, 0.495, 0.169, 0.086]
sel   = [1.000, 0.991, 0.889, 0.705, 0.343, 0.172]
fig, ax = plt.subplots(figsize=(6.2, 4.4))
ax.plot(rates, base, "o-", color="#888888", lw=1.8, ms=6,
        label="Baseline (ε=0.35, b=1.5, Rₚ=7µm)")
ax.plot(rates, sel, "s-", color="#3b7dd8", lw=1.8, ms=6,
        label="Optimized (ε=0.33, b=1.04, Rₚ=2.1µm)")
ax.set_xlabel("C-rate"); ax.set_ylabel("Capacity retention (Q$_C$/Q$_{0.5C}$)")
ax.set_title("Rate capability at matched areal loading (DFN)")
ax.legend(frameon=False, fontsize=9); ax.grid(alpha=0.25)
plt.tight_layout(); plt.savefig("fig4_rate_capability.png", dpi=200, bbox_inches="tight")
plt.show()

print("Done. Saved: fig1_shap_importance, fig2_pareto_front, "
      "fig3_spme_vs_dfn (if v2 present), fig4_rate_capability  → ")
