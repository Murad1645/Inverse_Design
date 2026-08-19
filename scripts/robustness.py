import numpy as np, pandas as pd, pybamm
pybamm.set_logging_level("ERROR")
param=pybamm.ParameterValues("Chen2020"); dfn=pybamm.lithium_ion.DFN()
def upd(eps,b,Rp,L):
    p=param.copy(); p.update({"Negative electrode porosity":eps,
      "Negative electrode Bruggeman coefficient (electrolyte)":b,
      "Negative electrode Bruggeman coefficient (electrode)":b,
      "Negative particle radius [m]":Rp*1e-6,"Negative electrode thickness [m]":L*1e-6},check_already_exists=False)
    return p
def cap_at(d,Crate):
    p=upd(*d)
    try:
        sim=pybamm.Simulation(dfn,parameter_values=p,experiment=pybamm.Experiment([f"Discharge at {Crate}C until 2.5 V"]))
        sol=sim.solve(); return float(sol["Discharge capacity [A.h]"].data[-1])
    except Exception: return np.nan

BASE=(0.35,1.5,7.0,100.0)
SEL =(0.33,1.037,2.059,97.054)
rates=[0.5,1,2,3,4,5]
cb={C:cap_at(BASE,C) for C in rates}
cs={C:cap_at(SEL,C) for C in rates}
q05b, q05s = cb[0.5], cs[0.5]
print("Crate | Baseline Cap(Ah)  Qret | Selected Cap(Ah)  Qret | Cap gain")
print("-"*70)
for C in rates:
    qrb=cb[C]/q05b; qrs=cs[C]/q05s
    print(f" {C:>3}C |   {cb[C]:6.3f}        {qrb:5.3f} |   {cs[C]:6.3f}        {qrs:5.3f} | {cs[C]/cb[C]:4.2f}x")
