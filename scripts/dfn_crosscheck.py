import numpy as np, pybamm, time
pybamm.set_logging_level("ERROR")
param = pybamm.ParameterValues("Chen2020")

def update_params(p0, eps,b,Rp,L):
    p=p0.copy()
    p.update({"Negative electrode porosity":eps,
        "Negative electrode Bruggeman coefficient (electrolyte)":b,
        "Negative electrode Bruggeman coefficient (electrode)":b,
        "Negative particle radius [m]":Rp*1e-6,
        "Negative electrode thickness [m]":L*1e-6},check_already_exists=False)
    return p

def evaluate(model, eps,b,Rp,L):
    p=update_params(param,eps,b,Rp,L); arrs={}
    for C in (0.5,3.0):
        exp=pybamm.Experiment([f"Discharge at {C}C until 2.5 V"])
        sim=pybamm.Simulation(model,parameter_values=p,experiment=exp)
        try:
            sol=sim.solve(); Q=sol["Discharge capacity [A.h]"].data; V=sol["Terminal voltage [V]"].data
            if len(Q)<2 or Q[-1]<=0: return None
            arrs[C]=(Q,V)
        except Exception as e:
            return None
    Q05,V05=arrs[0.5]; Q3,V3=arrs[3.0]
    Qr=float(Q3[-1]/Q05[-1]); qref=0.5*Q3[-1]
    dV=float(np.interp(qref,Q05,V05)-np.interp(qref,Q3,V3))
    return dict(Q05=float(Q05[-1]),Q3=float(Q3[-1]),Q_ratio=Qr,dV=dV)

spme=pybamm.lithium_ion.SPMe()
dfn =pybamm.lithium_ion.DFN()

designs={"BASELINE (0.35,1.5,7,100)":(0.35,1.5,7.0,100.0),
         "SELECTED (0.281,1.03,2.94,90.4)":(0.281,1.030,2.94,90.4)}

for name,(eps,b,Rp,L) in designs.items():
    print("="*64); print(name); print("="*64)
    for label,model in [("SPMe",spme),("DFN ",dfn)]:
        t0=time.time(); r=evaluate(model,eps,b,Rp,L); dt=time.time()-t0
        if r is None:
            print(f"  {label}: SOLVER FAILED ({dt:.1f}s)")
        else:
            print(f"  {label}: Q_ratio={r['Q_ratio']:.3f}  Q3={r['Q3']:.3f}Ah  Q05={r['Q05']:.3f}Ah  dV={r['dV']:.3f}V  ({dt:.1f}s)")
    print()
