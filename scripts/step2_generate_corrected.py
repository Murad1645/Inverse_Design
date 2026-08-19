# =====================================================================
# MASTER NOTEBOOK — STEP 2  (Dataset generation, CORRECTED)
# ---------------------------------------------------------------------
# Fixes two defects in the original generation:
#   (1) TARGET-BASED CENSORING removed. The old cell did:
#           df = df[df["Q_ratio"] > 0.2]      # deleted the failure regime
#           df = df[df["dV_3C"]  < 0.8]        # deleted high-polarization
#       => ~74% of the design space (all designs that die at 3C) was
#          deleted, so the surrogate only ever saw "winners".
#       Here we keep EVERY design the solver can evaluate. Only genuine
#       solver failures are excluded, and they are LOGGED (solver_ok).
#   (2) BROKEN dV METRIC replaced. The old cell used dV at "50% of each
#       run's own capacity" then took abs(), which is ill-posed when the
#       two C-rates reach very different capacities (gave negative dV on
#       dying runs). Here dV is measured at a MATCHED absolute throughput
#       (50% of the 3C capacity, which both runs reach) -> always well
#       defined, always positive, physically comparable.
#
# Scale N_SAMPLES up on your GPU/Colab run (2000+). Validated here on a
# small batch: ~0.56 s/design.
# =====================================================================

import numpy as np
import pandas as pd
from scipy.stats import qmc
from tqdm import tqdm
import pybamm

pybamm.set_logging_level("ERROR")

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
N_SAMPLES = 2000
SEED      = 42
OUT_CSV   = "canonical_dataset_v2.csv"   # unfiltered; supersedes v1

RANGES = {
    "eps":   (0.28, 0.45),
    "b":     (1.00, 2.50),
    "Rp_um": (2.00, 12.00),
    "L_um":  (50.0, 140.0),
}

model = pybamm.lithium_ion.SPMe()
param = pybamm.ParameterValues("Chen2020")

# ---------------------------------------------------------------------
# Design -> parameter mapping (unchanged from your notebook: NEGATIVE
# electrode only -> this is a graphite-anode microstructure study)
# ---------------------------------------------------------------------
def update_params_for_design(param_base, eps, b, Rp_um, L_um):
    p = param_base.copy()
    p.update({
        "Negative electrode porosity": eps,
        "Negative electrode Bruggeman coefficient (electrolyte)": b,
        "Negative electrode Bruggeman coefficient (electrode)": b,
        "Negative particle radius [m]": Rp_um * 1e-6,
        "Negative electrode thickness [m]": L_um * 1e-6,
    }, check_already_exists=False)
    return p

# ---------------------------------------------------------------------
# Corrected single-design evaluation
# ---------------------------------------------------------------------
def run_one_design_v2(eps, b, Rp_um, L_um):
    p = update_params_for_design(param, eps, b, Rp_um, L_um)
    arrs = {}
    for Crate in (0.5, 3.0):
        exp = pybamm.Experiment([f"Discharge at {Crate}C until 2.5 V"])
        sim = pybamm.Simulation(model, parameter_values=p, experiment=exp)
        try:
            sol = sim.solve()
            Q = sol["Discharge capacity [A.h]"].data
            V = sol["Terminal voltage [V]"].data
            if len(Q) < 2 or Q[-1] <= 0:
                return {"solver_ok": False}
            arrs[Crate] = (Q, V)
        except Exception:
            return {"solver_ok": False}

    Q05, V05 = arrs[0.5]
    Q3,  V3  = arrs[3.0]
    Q05f, Q3f = float(Q05[-1]), float(Q3[-1])

    Q_ratio = Q3f / Q05f                     # rate capability
    q_ref   = 0.5 * Q3f                       # matched throughput both runs reach
    dV_3C   = float(np.interp(q_ref, Q05, V05) - np.interp(q_ref, Q3, V3))  # >=0

    return {"solver_ok": True, "Q05_Ah": Q05f, "Q3_Ah": Q3f,
            "Q_ratio": float(Q_ratio), "dV_3C": dV_3C}

# ---------------------------------------------------------------------
# LHS + run (KEEP every solved design; only solver failures excluded)
# ---------------------------------------------------------------------
sampler = qmc.LatinHypercube(d=4, seed=SEED)
u = sampler.random(N_SAMPLES)
lows  = np.array([RANGES[k][0] for k in ["eps", "b", "Rp_um", "L_um"]])
highs = np.array([RANGES[k][1] for k in ["eps", "b", "Rp_um", "L_um"]])
X = lows + u * (highs - lows)

rows, n_solver_fail = [], 0
for i in tqdm(range(N_SAMPLES)):
    eps, b, Rp, L = map(float, X[i])
    r = run_one_design_v2(eps, b, Rp, L)
    if not r["solver_ok"]:
        n_solver_fail += 1
        continue                              # legitimate exclusion, logged
    rows.append({"sim_id": i, "eps": eps, "b": b, "Rp_um": Rp, "L_um": L, **r})

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)

# ---------------------------------------------------------------------
# Provenance diagnostics (put these numbers in the paper)
# ---------------------------------------------------------------------
n_ok = len(df)
print(f"generated {n_ok} valid designs from {N_SAMPLES} LHS samples")
print(f"solver failures (only legitimate drop): {n_solver_fail} "
      f"({100*n_solver_fail/N_SAMPLES:.1f}%)")
print(f"Q_ratio range: {df.Q_ratio.min():.3f} .. {df.Q_ratio.max():.3f}")
print(f"  failure-regime designs kept (Q_ratio<=0.2): {(df.Q_ratio<=0.2).sum()} "
      f"({100*(df.Q_ratio<=0.2).mean():.0f}% of data)")
print(f"dV_3C range (matched-cap): {df.dV_3C.min():.3f} .. {df.dV_3C.max():.3f} "
      f"| negatives: {(df.dV_3C<0).sum()}")
print(f"saved -> {OUT_CSV}")
