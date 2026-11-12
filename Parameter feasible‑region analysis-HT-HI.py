# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 09:45:05 2022

@author: tinghuang
"""

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#%% ===================== 1. light dark function =====================
def lightdark(t):
    if np.mod(t,24)>=12:
        L = 0
        D = 1
    else:
        L = 1
        D = 0
    return L,D
# --------------------------
#%% ODE system
def ode_system(y, t, p):
    MCL, CL, MP97, P97, MP51, P51, MEL, EL, P, MLhcb1, Lhcb1, MpsbA, psbA, MRbcS1, RbcS1, MatpA, atpA  = y
    v1, v1L, v2, v2L, v3, v4, v4L, v5, v6, v7, v8, p1, p1L, p2, p3, p4, p5, p6, p7, p8, pL, d1, d2L, d2D, d3L, d3D, d4L, d4D, dL, d5, d6, d7, d8, k1L, k1D, k2, k3, k4, k5, k6, k7, k8, K0, K1, K2, K3, K4, K5, K6, K7, K8, K9, K10, K11, K12, K13, K14, K15, K16, K17, v3L, deltL, lam, LTH, deltTp, deltTm, n, Q  = p
    L,D = lightdark(t)
    
    I = 450
    g = I**n/(LTH**n+I**n)
    Tref = 25
    T = 37
    tot = (T-Tref)/Tref
    deltT = -deltTm+(deltTm+deltTp)*g
    phi = 1/(1+lam*g)
    gamaL=-deltL*tot*phi
    gamaT=deltT *tot
    
    dL = dL*Q**((T-Tref)/10)

    pL = pL*Q**((T-Tref)/10)
    
    dMCLdt =(v1+v1L*P*g+gamaL)/(1+(P97/K1)**2+(P51/K2)**2+(CL/K0)**2)-(k1L*L+k1D*D)*MCL
    dCLdt = (p1+p1L*L)*MCL-d1*CL
    dMP97dt = (v2L*L*P+v2)/(1+(EL/K3)**2+(CL/K4)**2)-k2*MP97
    dP97dt = p2*MP97-(d2D*D+d2L*L)*P97
    dMP51dt = (v3+v3L*P*g+gamaT)/(1+(CL/K6)**2+(P51/K5)**2)-k3*MP51
    dP51dt = p3*MP51-(d3D*D+d3L*L)*P51
    dMELdt = (v4+v4L*L)/(1+(CL/K7)**2+(P51/K8)**2)*(P97/K9)**2/(1+(P97/K9)**2)-k4*MEL
    dELdt = p4*MEL-(d4D*D+d4L*L)*EL
    dPdt = pL*(1-P)*D-dL*P*L 
    dMLhcb1dt = v5*(CL/K10)**2/(1+(CL/K10)**2)/(1+(P51/K11)**2)-k5*MLhcb1
    dLhcb1dt = p5*MLhcb1-d5*Lhcb1
    dMpsbAdt = v6/(1+(CL/K12)**2+(P51/K13)**2)-k6*MpsbA
    dpsbAdt = p6*MpsbA-d6*psbA
    dMRbcS1dt = v7/(1+(CL/K14)**2+(P97/K15)**2)-k7*MRbcS1
    dRbcS1dt = p7*MRbcS1-d7*RbcS1
    dMatpAdt = v8/(1+(CL/K16)**2+(P97/K17)**2)-k8*MatpA
    datpAdt = p8*MatpA-d8*atpA
    return [dMCLdt, dCLdt, dMP97dt, dP97dt, dMP51dt, dP51dt, dMELdt, dELdt, dPdt, dMLhcb1dt, dLhcb1dt, dMpsbAdt, dpsbAdt, dMRbcS1dt, dRbcS1dt, dMatpAdt, datpAdt]

def calc_metrics(t, y):
    # y: (Ntime, Nvar)
    mask = t >= 144
    if np.sum(mask) == 0:
        raise ValueError("No data in 120~144h window!")
    t_ss = t[mask]
    y_ss = y[mask, :]  # shape [Nss, Nvar]
    
    var_names = [
        "cca1",
        "prr9",
        "toc1",
        "elf4",
        "lhcb",
        "psba",
        "rbcs",
        "atpa"
    ]
    var_indices = [0, 2, 4, 6, 9, 11, 13, 15]
    res = {}
    for name, idx in zip(var_names, var_indices):
        sig = y_ss[:, idx]   # ========== 修复点：取该基因在稳态窗口的时序！==========
        sig_max = np.max(sig)
        sig_min = np.min(sig)
        sig_amp = sig_max - sig_min
        idx_max = np.argmax(sig)
        t_peak = t_ss[idx_max]
        sig_phase = (t_peak - 144) % 24   # 相位：相对于120h，0~24h
        res[f"{name}_amp"] = sig_amp
        res[f"{name}_phase"] = sig_phase
    return res




#%% =========constrain condition=========  control the phase and amplitude
# CCA1
PHASE_CCA1_MIN = -3
PHASE_CCA1_MAX = 3
AMP_CCA1_MIN = 0.1
# PRR9
PHASE_PRR9_MIN = 6
PHASE_PRR9_MAX = 10
AMP_PRR9_MIN = 0.1
# TOC1
PHASE_TOC1_MIN = 10
PHASE_TOC1_MAX = 16
AMP_TOC1_MIN = 0.1
# ELF4
PHASE_ELF4_MIN = 10
PHASE_ELF4_MAX = 16
AMP_ELF4_MIN = 0.05
# Lhcb
PHASE_LHCB_MIN = 4
PHASE_LHCB_MAX = 10
AMP_LHCB_MIN = 0.1
# psbA
PHASE_PSBA_MIN = 8
PHASE_PSBA_MAX = 14
AMP_PSBA_MIN = 0.1
# RbcS
PHASE_RBCS_MIN = 16
PHASE_RBCS_MAX = 20
AMP_RBCS_MIN = 0.1
# AtpA
PHASE_ATPA_MIN = 18
PHASE_ATPA_MAX = 23
AMP_ATPA_MIN = 0.1

def check_constraint(t, y):
    met = calc_metrics(t, y)
    # check output
    # print(met)
    ok_cca1  = (PHASE_CCA1_MIN <= met["cca1_phase"] <= PHASE_CCA1_MAX or 24-PHASE_CCA1_MAX <= met["cca1_phase"] <= 24) and (met["cca1_amp"] > AMP_CCA1_MIN) 
    ok_prr9  = (PHASE_PRR9_MIN <= met["prr9_phase"] <= PHASE_PRR9_MAX) and (met["prr9_amp"] > AMP_PRR9_MIN)
    ok_toc1  = (PHASE_TOC1_MIN <= met["toc1_phase"] <= PHASE_TOC1_MAX) and (met["toc1_amp"] > AMP_TOC1_MIN)
    ok_elf4  = (PHASE_ELF4_MIN <= met["elf4_phase"] <= PHASE_ELF4_MAX) and (met["elf4_amp"] > AMP_ELF4_MIN)
    ok_lhcb  = (PHASE_LHCB_MIN <= met["lhcb_phase"] <= PHASE_LHCB_MAX) and (met["lhcb_amp"] > AMP_LHCB_MIN)
    ok_psba  = (PHASE_PSBA_MIN <= met["psba_phase"] <= PHASE_PSBA_MAX) and (met["psba_amp"] > AMP_PSBA_MIN)
    ok_rbcs  = (PHASE_RBCS_MIN <= met["rbcs_phase"] <= PHASE_RBCS_MAX) and (met["rbcs_amp"] > AMP_RBCS_MIN)
    ok_atpa  = (PHASE_ATPA_MIN <= met["atpa_phase"] <= PHASE_ATPA_MAX) and (met["atpa_amp"] > AMP_ATPA_MIN)
    ok = ok_cca1 and ok_prr9 and ok_toc1 and ok_elf4 and ok_lhcb and ok_psba and ok_rbcs and ok_atpa
    # ok = ok_cca1 and  ok_toc1 
    
    return ok, met



def scan_single_param(p_base, idx_param, y0,
                      search_range=(0.01, 5.0), num_points=300):
    
    p0 = p_base[idx_param]
    test_factors = np.linspace(search_range[0], search_range[1], num_points)
    valid_p = []
    t_eval = np.arange(0,168,0.01)
    for fac in test_factors:
        p_test = np.copy(p_base)
        p_test[idx_param] = p0 * fac
        try:
            y = odeint(ode_system, y0, t_eval, args=(p_test,))
        except Exception:
            continue
        ok,_ = check_constraint(t_eval, y)
        if ok:
            valid_p.append(p0*fac)
    if len(valid_p)==0:
        return np.nan, np.nan
    return min(valid_p), max(valid_p)
#%% =================main====================
p_baseline = np.array([2.7177,0.7601,0.3098,2.2044,5.0094,0.9755,2.7704,2.4447,14.183,5.8215,1.6568,4.8964,0.5554,3.5756,1.1509,3.5684,0.6515,2.6233,0.5447,2.3698,2.0831,0.9166,0.4136,0.7106,1.2734,0.2786,0.1084,0.3042,0.2348,1.2596,0.5022,1.1368,1.5255,0.7036,0.6009,0.9622,5.2985,4.9498,0.8139,0.5654,0.665,0.887,8.8257,0.5281,3.8125,5.2146,3.6288,6.0682,0.6945,0.2286,2.7464,2.5992,0.2938,0.3154,1.473,1.0154,1.6716,1.0154,8.9283,1.9154,4.45745,0.4623,1.0477,234.6663,2.0486,1.4017,4,0.8828 ])

y0_init = [1.52318295e-01,3.39972549e-01,5.95471718e-01,2.15006756e+00,1.35930447e+00,1.09418276e+00,3.15492116e-01,8.46288136e-01,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]

# =========test the basal parameters=========
print("===== basal parameter verification =====")

sol_test_t = np.arange(0, 168,0.01)
sol_test_y = odeint(ode_system, y0_init, sol_test_t, args=(p_baseline,))

met_test = calc_metrics(sol_test_t, sol_test_y)

print(met_test)

ok_test,_ = check_constraint(sol_test_t, sol_test_y)

print("Test the basal parameters (satisfy the constraints)：", ok_test)

# =================================

param_names = [
    "v1","v1L","v2","v2L","v3","v4","v4L","v5","v6","v7","v8",
    "p1","p1L","p2","p3","p4","p5","p6","p7","p8","pL",
    "d1","d2L","d2D","d3L","d3D","d4L","d4D","dL","d5","d6","d7","d8",
    "k1L","k1D","k2","k3","k4","k5","k6","k7","k8",
    "K0","K1","K2","K3","K4","K5","K6","K7","K8","K9","K10","K11","K12","K13","K14","K15","K16","K17",
    "v3L", "deltL", "lam", "LTH", "deltTp", "deltTm", "n", "Q"
]
res_list = []
for i,par_name in enumerate(param_names):
    print(f"scanning parameter: {par_name}")
    p_low, p_high = scan_single_param(p_baseline, idx_param=i, y0=y0_init)
    p_ref = p_baseline[i]
    res_list.append({
        "param":par_name,
        "p_ref": p_ref,
        "p_low":p_low,
        "p_high":p_high,
        "fold_low": p_low/p_ref if not np.isnan(p_low) else np.nan,
        "fold_high": p_high/p_ref if not np.isnan(p_high) else np.nan
    })
df_scan = pd.DataFrame(res_list)
df_scan.to_csv("parameter_feasible_range_phase_constrain_HT_HI.csv",index=False)
print(df_scan.round(3))

plt.rcParams["font.family"] = "Times New Roman"
fig, ax = plt.subplots(figsize=(10, 12), dpi=120)
df_valid = df_scan.dropna().copy()
y_pos = np.arange(len(df_valid))
base_fold = 1.0

# the feasible interval
fold_low  = df_valid["fold_low"].values
fold_high = df_valid["fold_high"].values

bar_height = 0.35

# ========== distinguish the sensitive parameters ==========
for i, y in enumerate(y_pos):
    x0 = fold_low[i]
    x1 = fold_high[i]
    width = x1 - x0
   
    if width < 1.2:  # defined by self
        fill_color = "#1f4e96"
        fill_alpha = 0.85
    else:
        fill_color = "#4267b2"
        fill_alpha = 0.65
    ax.fill_betweenx([y-bar_height/2, y+bar_height/2], x0, x1, 
                     color=fill_color, alpha=fill_alpha)

# ==========the optimal fit point ==========
ax.scatter(np.full(len(df_valid), base_fold), y_pos, 
           color="#0f2b66", zorder=10, s=45)
ax.axvline(x=1, linestyle="--", color="#444444", label="Best‑fit value (1.0 fold)")

ax.set_yticks(y_pos)
ax.set_yticklabels(df_valid["param"], fontsize=11)
ax.set_xlabel("Relative parameter value (fold of best‑fit)", fontsize=14)
ax.set_ylabel("Model parameter", fontsize=14)
ax.legend(fontsize=11)



# plt.tight_layout()
plt.savefig("parameter_feasible_region_HT_HI.pdf", bbox_inches="tight")
plt.savefig("parameter_feasible_region_HT_HI.tif", bbox_inches="tight", dpi=500)
# plt.show()

