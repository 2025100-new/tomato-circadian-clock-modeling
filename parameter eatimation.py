# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 10:57:49 2026

@author: tinghuang
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares, differential_evolution

# ===================== 1. light/dark cycles =====================
def lightdark(t):
    if np.mod(t,24)>=12:
        L = 0
        D = 1
    else:
        L = 1
        D = 0
    return L,D

# ===================== 2. ODE system =====================
def ode_system(t, y, p):
    LHYm, LHY, TOC1m, TOC1, P = y
    vLB, vTB, vLI, vTI, v1, v2, p1, p2, p3, d1, d2, d3, k1,  k2, K1, K2, K3, K4, deltL, lam, LTH, deltTp, deltTm, Q, n = p
    L,D = lightdark(t)
    I = 450
    # n=4
    g = I**n/(LTH**n+I**n)
    Tref = 25
    T = 25
    
    tot = (T-Tref)/Tref
    
    deltT = -deltTm+(deltTm+deltTp)*g
    
    phi = 1/(1+lam*g)
    
    gamaL=-deltL*tot*phi
    gamaT=deltT *tot
    
    dL = d1*Q**((T-Tref)/10)
    dT = d2*Q**((T-Tref)/10)
    
    dLHYmdt =(vLB+vLI*g*P+gamaL)*LHY**2/(K1**2+LHY**2)*TOC1**2/(K2**2+TOC1**2)-k1*LHYm
    dLHYdt = p1*LHYm-dL*LHY
    dTOC1mdt = (vTB+vTI*g*P++gamaT)*K3**2/(K3**2+LHY**2)*TOC1**2/(K4**2+TOC1**2)-k2*TOC1m
    dTOC1dt = p2*TOC1m-dT*TOC1
    dPdt = p3*(1-P)*D - d3*P*L 
    return [dLHYmdt, dLHYdt, dTOC1mdt, dTOC1dt, dPdt]

# ===================== 3. model simulation =====================
def simulate_relax(t_obs, y0, params, relax_time=72):
    sol_heat = solve_ivp(fun=lambda t,y: ode_system(t,y,params),
                    t_span=(0, relax_time),
                    y0=y0,
                    method="BDF")
    y_relaxed = sol_heat.y[:,-1]
    
    t_start = np.min(t_obs)
    t_end = np.max(t_obs)
    sol = solve_ivp(fun=lambda t,y: ode_system(t,y,params),
                    t_span=(t_start, t_end),
                    y0=y_relaxed,
                    t_eval=t_obs,
                    method="BDF")
    return sol.y.T

# ===================== 4. Residual function（LHYm、TOC1m） =====================
def residual_func(p_guess, t_obs, y_obs_target, y0):
    y_sim = simulate_relax(t_obs, y0, p_guess)
    sim_target = np.column_stack([y_sim[:,0], y_sim[:,2]])
    resid = (sim_target - y_obs_target).ravel()
    return resid

# ===================== 5. Data loading =====================
t_obs = np.arange(0, 49, 3)

# ===== experimental data =====
df = pd.read_csv("circadian_data-CK-HI.csv", header=0)

y_obs_target = df[["LHYm", "TOC1m"]].values
# ===================== 6. Boundary Definition =====================
# differential_evolution    vLB, vTB, vLI, vTI, v1, v2, p1, p2, p3, d1, d2, d3, k1,  k2, K1, K2, K3, K4, deltL, lam, LTH, deltTp, deltTm, Q
bounds_list = [
    (0.01, 5.0),   # vLB
    (0.01, 5.0),   # vTB
    (0.01, 5.0),   # vLI
    (0.01, 5.0),   # vTI
    (0.01, 5.0),   # v1
    (0.01, 5.0),   # v2
    (0.10, 4.2),   # p1
    (0.10, 4.2),   # p2
    (0.02, 3.5),   # p3
    (0.02, 1.4),   # d1
    (0.02, 2.4),   # d2
    (0.02, 1.4),   # d3
    (0.05, 2.5),   # k1
    (0.05, 2.5),   # k2
    (0.20, 2.0),   # K1
    (0.20, 2.0),   # K2
    (0.20, 2.0),   # K3
    (0.20, 2.0),   # K4
    (0.05, 2.5),   # deltL
    (0.05, 2.5),   # lam
    (180, 320),   # LTH
    (0.05, 2.5),   # deltTp
    (0.05, 2.5),   # deltTm
    (0.05, 4.5),   # Q
    (1, 4),  #n
]
# Convert to the format required by least_squares (lb_array, ub_array)
lb = np.array([b[0] for b in bounds_list])
ub = np.array([b[1] for b in bounds_list])
ls_bounds = (lb, ub)

y0 = np.array([0.8, 1.2, 0.25, 1.0, 0.4])

# ===================== 7. Two-stage fitting =====================
print("=====Stage 1: Differential Evolution Global Coarse Search=====")
def cost_global(p):
    return np.sum(np.square(residual_func(p, t_obs, y_obs_target, y0)))

de_result = differential_evolution(cost_global, bounds=bounds_list, maxiter=50, popsize=8)
p_global = de_result.x
print("Global search parameters：")
print(p_global)

print("\n=====Stage 2: Least Squares Fine Optimization =====")
fit_result = least_squares(
    fun=residual_func,
    x0=p_global,
    args=(t_obs, y_obs_target, y0),
    bounds=ls_bounds,
    max_nfev=15000,
    loss="linear"
)

p_fit = fit_result.x
print("Convergent state success:", fit_result.success)
param_names = ["vLB","vTB","vLI","vTI","v1","v2","p1","p2","p3","d1","d2","d3","k1","k2","K1","K2","K3","K4", "deltL","lam","LTH","deltTp","deltTm","Q","n"]
for name,val in zip(param_names, p_fit):
    print(f"{name:6s} = {val:.4f}")

# ===================== 8. Plot =====================
y_all_fit = simulate_relax(t_obs, y0, p_fit)
lhydm_fit = y_all_fit[:,0]
toc1m_fit = y_all_fit[:,2]

fig, (ax1, ax2) = plt.subplots(2,1,figsize=(10,9), sharex=True)
ax1.plot(t_obs, y_obs_target[:,0], "o", color="#1f77b4", markersize=7, label="Observed LHYm")
ax1.plot(t_obs, lhydm_fit, "-", color="#1f77b4", lw=2.5, label="Model fit LHYm")
ax1.set_ylabel("LHYm", fontsize=15)
ax1.legend(fontsize=12)
ax1.grid(alpha=0.25)

ax2.plot(t_obs, y_obs_target[:,1], "o", color="#d62728", markersize=7, label="Observed TOC1m")
ax2.plot(t_obs, toc1m_fit, "-", color="#d62728", lw=2.5, label="Model fit TOC1m")
ax2.set_xlabel("Time (h)", fontsize=15)
ax2.set_ylabel("TOC1m", fontsize=15)
ax2.legend(fontsize=12)
ax2.grid(alpha=0.25)

plt.tight_layout()
plt.savefig("circadian_fit_optimized.tif", dpi=300, bbox_inches="tight")
plt.show()