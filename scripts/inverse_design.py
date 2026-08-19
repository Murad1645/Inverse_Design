import numpy as np, pandas as pd, pybamm
from xgboost import XGBRegressor
pybamm.set_logging_level("ERROR")

df = pd.read_csv("canonical_dataset_v3_dfn.csv")
if "solver_ok" in df: df = df[df["solver_ok"]==True]
FEAT=["eps","b","Rp_um","L_um"]; X=df[FEAT].values
def xgb(): return XGBRegressor(n_estimators=800,learning_rate=0.03,max_depth=4,subsample=0.9,
    colsample_bytree=0.9,reg_lambda=1.0,random_state=42,n_jobs=-1,tree_method="hist")
mQ=xgb(); mQ.fit(X, df.Q_ratio.values)
mD=xgb(); mD.fit(X, df.dV_3C.values)

# ---- baseline + loading target ----
BASE=dict(eps=0.35,b=1.5,Rp_um=7.0,L_um=100.0)
S0=(1-BASE["eps"])*BASE["L_um"]              # = 65.0  loading proxy
RANGES={"eps":(0.28,0.45),"b":(1.0,2.5),"Rp_um":(2.0,12.0),"L_um":(50.0,140.0)}

# ---- 1) generate candidates, 2) enforce loading-match FIRST ----
rng=np.random.default_rng(0); N=200000
eps=rng.uniform(*RANGES["eps"],N); b=rng.uniform(*RANGES["b"],N); Rp=rng.uniform(*RANGES["Rp_um"],N)
L=S0/(1-eps)                                  # thickness pinned so (1-eps)L = S0 exactly
m=(L>=RANGES["L_um"][0])&(L<=RANGES["L_um"][1])
C=np.column_stack([eps[m],b[m],Rp[m],L[m]])
qp=mQ.predict(C); dp=mD.predict(C)
print(f"loading-matched candidates in-bounds: {len(C)}  (S0={S0:.1f})")

# ---- 3) Pareto front (max Q, min dV) ----
def pareto(q,d):
    o=np.argsort(-q); front=[]; best=np.inf
    for i in o:
        if d[i]<best-1e-9: front.append(i); best=d[i]
    return np.array(front)
pf=pareto(qp,dp)
# ---- 4) representative knee point (max normalized q - normalized d) ----
qn=(qp[pf]-qp[pf].min())/(np.ptp(qp[pf])); dn=(dp[pf]-dp[pf].min())/(np.ptp(dp[pf]))
knee=pf[np.argmax(qn-dn)]
sel=dict(eps=C[knee,0],b=C[knee,1],Rp_um=C[knee,2],L_um=C[knee,3])
print(f"Pareto points: {len(pf)}")
print("SELECTED (surrogate):", {k:round(v,3) for k,v in sel.items()},
      f"| surrogate Q={qp[knee]:.3f} dV={dp[knee]:.3f}")

# ---- 5) DFN re-validation of baseline + selected ----
param=pybamm.ParameterValues("Chen2020"); dfn=pybamm.lithium_ion.DFN()
def upd(eps,b,Rp,L):
    p=param.copy(); p.update({"Negative electrode porosity":eps,
      "Negative electrode Bruggeman coefficient (electrolyte)":b,
      "Negative electrode Bruggeman coefficient (electrode)":b,
      "Negative particle radius [m]":Rp*1e-6,"Negative electrode thickness [m]":L*1e-6},check_already_exists=False)
    return p
def dfn_eval(d):
    p=upd(d["eps"],d["b"],d["Rp_um"],d["L_um"]); a={}
    for Cr in (0.5,3.0):
        sim=pybamm.Simulation(dfn,parameter_values=p,experiment=pybamm.Experiment([f"Discharge at {Cr}C until 2.5 V"]))
        sol=sim.solve(); a[Cr]=(sol["Discharge capacity [A.h]"].data,sol["Terminal voltage [V]"].data)
    Q05,V05=a[0.5]; Q3,V3=a[3.0]; qref=0.5*Q3[-1]
    return dict(Q05=float(Q05[-1]),Q3=float(Q3[-1]),Q_ratio=float(Q3[-1]/Q05[-1]),
                dV=float(np.interp(qref,Q05,V05)-np.interp(qref,Q3,V3)))
rb=dfn_eval(BASE); rs=dfn_eval(sel)
print()
print("=== DFN RE-VALIDATION (loading-matched, S0=65) ===")
print(f"  BASELINE  { {k:round(v,3) for k,v in BASE.items()} }")
print(f"            Q_ratio={rb['Q_ratio']:.3f}  Q3={rb['Q3']:.3f}Ah  dV={rb['dV']:.3f}V")
print(f"  SELECTED  { {k:round(v,3) for k,v in sel.items()} }")
print(f"            Q_ratio={rs['Q_ratio']:.3f}  Q3={rs['Q3']:.3f}Ah  dV={rs['dV']:.3f}V")
print()
print(f"  surrogate-vs-DFN check on selected: Q {qp[knee]:.3f} vs {rs['Q_ratio']:.3f} | dV {dp[knee]:.3f} vs {rs['dV']:.3f}")
print(f"  >> HONEST IMPROVEMENT: rate {rs['Q_ratio']/rb['Q_ratio']:.2f}x | "
      f"3C cap {rs['Q3']/rb['Q3']:.2f}x | polarization {100*(1-rs['dV']/rb['dV']):.0f}% lower")
