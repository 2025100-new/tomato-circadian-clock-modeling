# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 17:39:42 2026

@author: Ting Huang
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ===================== 1. light function =====================


def lightdark(t):
    L =1
    D = 0
    return L,D

# ===================== 2. ODE system =====================
def ode_system_env(t, y, p, I_input, T_input):
    LHYm, LHY, TOC1m, TOC1, P = y
    vLB, vTB, vLI, vTI, p1, p2, p3, d1, d2, d3, k1,  k2, K1, K2, K3, K4, deltL, lam, LTH, deltTp, deltTm, Q, n = p
    L,D = lightdark(t)
    I_input = 450
    g = I_input**n/(LTH**n+I_input**n)
    Tref = 25
    T_input = 25
    
    tot = (T_input-Tref)/Tref
    
    deltT = -deltTm+(deltTm+deltTp)*g
    
    phi = 1/(1+lam*g)
    
    gamaL=-deltL*tot*phi
    gamaT=deltT *tot
    
    dL = d1*Q**((T_input-Tref)/10)
    dT = d2*Q**((T_input-Tref)/10)
    
    dLHYmdt = (vLB+vLI*g*P+gamaL)*LHY**2/(K1**2+LHY**2)*TOC1**2/(K2**2+TOC1**2)-k1*LHYm
    dLHYdt = p1*LHYm-dL*LHY
    dTOC1mdt = (vTB+vTI*g*P+gamaT)*K3**2/(K3**2+LHY**2)*TOC1**2/(K4**2+TOC1**2)-k2*TOC1m
    dTOC1dt = p2*TOC1m-dT*TOC1
    dPdt = p3*(1-P)*D - d3*P*L 
    return [dLHYmdt, dLHYdt, dTOC1mdt, dTOC1dt, dPdt]

def calc_metrics(t,y):
    cca1 = y[0,:]
    toc1 = y[2,:]
    cca1_max = np.max(cca1)
    cca1_min = np.min(cca1)
    cca1_amp = cca1_max - cca1_min
    idx_cca = np.argmax(cca1)
    cca1_phase = t[idx_cca]
    if cca1_phase == 0:
        cca1_phase = 24
    toc1_max = np.max(toc1)
    toc1_min = np.min(toc1)
    toc1_amp = toc1_max - toc1_min
    idx_toc1 = np.argmax(toc1)
    toc1_phase = t[idx_toc1]
    return {"cca1_amp":cca1_amp,"cca1_phase":cca1_phase,
            "toc1_amp":toc1_amp,"toc1_phase":toc1_phase}

def bifurcation_state_scan(p_baseline, param_idx, y0, I_input, T_input,
                           factor_min=0.05, factor_max=3.0, n_points=150):
   
    p0 = p_baseline[param_idx]
    factor_list = np.linspace(factor_min, factor_max, n_points)
    res = []
    for fac in factor_list:
        p_test = np.copy(p_baseline)
        p_test[param_idx] = p0 * fac

        sol = solve_ivp(ode_system_env, t_span=(0,96), y0=y0,
                        args=(p_test, I_input, T_input),
                        t_eval=np.arange(0,96,0.5), method="RK45")
        if not sol.success:
            res.append({"factor":fac, "x_max":np.nan, "x_min":np.nan})
            continue

       
        mask = sol.t >= 48
        y_ss = sol.y[:, mask]
        x = y_ss[0]   

        xmax = np.max(x)
        xmin = np.min(x)
        res.append({"factor":fac, "x_max":xmax, "x_min":xmin})
    return pd.DataFrame(res)



# ====================== main ======================
p_baseline = np.array([3.7411,0.7157,2.2860,2.8453,3.7164,3.3913,0.9381,0.2781,0.2097,0.3319,1.1783 ,1.9349,0.7309,1.6249,1.5391,0.9957,0.4623,1.0477, 274.6663, 2.0486,1.4017,0.8828,1.3785])

y0_init =np.array([0.5, 0.1, 0.85, 0.2,0.5])
I_env, T_env = 450, 25

param_names = [
    "vLB","vTB","vLI","vTI","p1","p2","p3",
    "d1","d2","d3","k1","k2","K1","K2","K3","K4",
    "deltL","lam","LTH","deltTp","deltTm","Q","n"
]

# ========= parameter selection =========
scan_idx = 14    # the parameter index
scan_name = param_names[scan_idx]

df_bif = bifurcation_state_scan(p_baseline, param_idx=scan_idx,
                                y0=y0_init, I_input=I_env, T_input=T_env,
                                factor_min=0.01, factor_max=20, n_points=100)

df_bif.to_csv(f"bifurcation_state_{scan_name}.csv", index=False)

# =================Plot====================
plt.rcParams["font.family"] = "Times New Roman"
fig, ax = plt.subplots(figsize=(9,7), dpi=150)

fac = df_bif["factor"]
xmax = df_bif["x_max"]
xmin = df_bif["x_min"]


ax.plot(fac, xmax, color="black", lw=1.2)
ax.plot(fac, xmin, color="black", lw=1.2)


ax.axvline(x=1.0, linestyle="--", c="gray", lw=1, label="Best‑fit parameter")

ax.set_xlabel(f"Parameter {scan_name} (fold of best‑fit)", fontsize=11)
ax.set_ylabel("CCA1 mRNA level", fontsize=11)
ax.tick_params(labelsize=9)
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(f"bifurcation_hopf_style_{scan_name}.pdf", bbox_inches="tight")
plt.savefig(f"bifurcation_hopf_style_{scan_name}.tif", bbox_inches="tight", dpi=500)
plt.show()