# Loading-Matched, Physics-Validated Machine-Learning Inverse Design of High-Rate Lithium-Ion Battery Electrodes

This repository contains the complete dataset and code to reproduce the results of the paper:

> **Loading-matched, physics-validated machine-learning inverse design of high-rate lithium-ion battery electrodes**
> Md. Murad Hossen
> Department of Materials Science and Engineering
> Corresponding author: mdmuradhossen1645@gmail.com
> *(Manuscript in preparation)*

The study develops a fair-comparison, physics-validated machine-learning framework for inverse design of a porous graphite negative electrode. Full pseudo-two-dimensional (Doyle–Fuller–Newman) simulations in [PyBaMM](https://www.pybamm.org) are used to build an unfiltered dataset; gradient-boosted and random-forest surrogates predict rate capability and polarization; inverse design is performed under an explicit areal-loading constraint; and the selected design is re-validated in the full physics model. Two methodological points are demonstrated: (i) a loading-matched constraint prevents apparent gains from reduced active-material loading, and (ii) a reduced-order model (SPMe) misranks the high-rate design space relative to full physics.

---

## Repository contents

All scripts and the dataset live in the same folder (flat layout), so the scripts find the data with no path configuration.

| File | Purpose |
|------|---------|
| `canonical_dataset_v3_dfn.csv` | **Main dataset** — 1993 valid DFN (full-physics) designs; basis of all results |
| `canonical_dataset_v2.csv` | Unfiltered SPMe dataset (optional; needed only for the SPMe-vs-DFN figure) |
| `step2_generate_corrected.py` | Generates the unfiltered dataset (SPMe → rerun as DFN); **slow** (see note) |
| `dfn_crosscheck.py` | Direct SPMe-vs-DFN comparison on identical designs |
| `surrogate_cv.py` | Trains XGBoost / random-forest surrogates; 5-fold cross-validation (R², RMSE) |
| `inverse_design.py` | Loading-matched Pareto optimization and knee-point design selection |
| `robustness.py` | 1C–5C rate-capability sweep of baseline vs optimized electrode |
| `shap_mech.py` | SHAP global feature importance (mechanistic interpretation) |
| `make_all_figures.py` | Regenerates the four data figures (SHAP, Pareto, SPMe-vs-DFN, rate capability) |
| `method_schematic.py` | Generates the workflow schematic (Figure 1) |
| `requirements.txt` | Python dependencies |

> **Figure numbering note.** File names and manuscript figure numbers differ:
> `fig0_method_schematic` = **Figure 1**, `fig3_spme_vs_dfn` = **Figure 2**,
> `fig2_pareto_front` = **Figure 3**, `fig4_rate_capability` = **Figure 4**,
> `fig1_shap_importance` = **Figure 5**.

---

## Installation

Requires **Python 3.9+**. Install dependencies:

```bash
pip install -r requirements.txt
```

Key packages (exact versions pinned in `requirements.txt`):

| Package | Version |
|---------|---------|
| pybamm | 26.8.0 |
| xgboost | 3.4.0 |
| scikit-learn | 1.6.1 |
| shap | 0.52.0 |
| scipy | 1.16.3 |
| numpy | 2.0.2 |
| pandas | 2.2.3 |
| matplotlib | 3.10.0 |

---

## How to reproduce the results

The dataset (`canonical_dataset_v3_dfn.csv`) is included, so you can reproduce every downstream result **without** rerunning the expensive simulations. Run the scripts from the repository folder:

```bash
# 1. Surrogate accuracy (Section 3.1)             → R² ≈ 0.99 (Q_ratio), 0.91 (ΔV)
python surrogate_cv.py

# 2. Model-fidelity comparison (Section 3.2)      → 72% vs 24% failure regime
python dfn_crosscheck.py

# 3. Loading-matched inverse design (Section 3.3) → selected knee-point design
python inverse_design.py

# 4. Robustness across C-rates (Section 3.4)      → 1C–5C sweep
python robustness.py

# 5. Mechanism via SHAP (Section 3.5)             → thickness & tortuosity dominant
python shap_mech.py

# 6. Regenerate figures
python make_all_figures.py       # Figures 2–5 (data figures)
python method_schematic.py       # Figure 1 (workflow schematic)
```

### Regenerating the dataset from scratch (optional)

To rebuild the dataset from the physics simulations rather than using the provided CSV:

```bash
python step2_generate_corrected.py
```

> **Note.** Full DFN (P2D) generation over the design space is computationally intensive — on the order of tens of minutes to hours depending on hardware (≈37 min in the environment used for this study). The provided `canonical_dataset_v3_dfn.csv` lets you skip this step.

---

## Design space and key results

Four macroscopic negative-electrode design variables are optimized:

| Variable | Symbol | Range |
|----------|--------|-------|
| Porosity | ε | 0.28–0.45 |
| Bruggeman exponent (tortuosity) | b | 1.0–2.5 |
| Particle radius | Rₚ | 2–12 µm |
| Electrode thickness | L | 50–140 µm |

At matched areal loading, the optimized electrode improves 3C rate capability by ≈1.4× and reduces polarization by ≈30%, with the surrogate-predicted optimum reproduced by full physics to within ≈1%.

---

## Citation

If you use this code or dataset, please cite the paper:

```bibtex
@article{Hossen_electrode_design,
  title   = {Loading-matched, physics-validated machine-learning inverse design of high-rate lithium-ion battery electrodes},
  author  = {Hossen, Md. Murad},
  journal = {[Journal]},
  year    = {[Year]},
  doi     = {[DOI]}
}
```

This work relies on **PyBaMM**; please also cite Sulzer et al. (2021), *Journal of Open Research Software* 9(1):14, and the Chen2020 parameter set (Chen et al., 2020, *J. Electrochem. Soc.* 167:080534).

---

## License

Released under the [MIT License](LICENSE). <!-- add a LICENSE file, or change to your preferred license -->
