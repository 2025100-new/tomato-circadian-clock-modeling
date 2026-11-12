# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 17:39:42 2026

@author: Ting Huang
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ===================== 1. 光暗周期函数 =====================
# def lightdark(t):
#     if np.mod(t,24)>=12:
#         L = 0
#         D = 1
#     else:
#         L = 1
#         D = 0
#     return L,D


def lightdark(t):
    L =1
    D = 0
    return L,D

# --------------------------
# 你的ODE函数，复用你原有
def ode_system_env(t, y, p, I_input, T_input):
    MCL, CL, MP97, P97, MP51, P51, MEL, EL, P, MLhcb1, Lhcb1, MpsbA, psbA, MRbcS1, RbcS1, MatpA, atpA  = y
    v1, v1L, v2, v2L, v3, v4, v4L, v5, v6, v7, v8, p1, p1L, p2, p3, p4, p5, p6, p7, p8, pL, d1, d2L, d2D, d3L, d3D, d4L, d4D, dL, d5, d6, d7, d8, k1L, k1D, k2, k3, k4, k5, k6, k7, k8, K0, K1, K2, K3, K4, K5, K6, K7, K8, K9, K10, K11, K12, K13, K14, K15, K16, K17, v3L, deltL, lam, LTH, deltTp, deltTm, n, Q  = p
    
    L,D = lightdark(t)
    
    g = I_input**n/(LTH**n+I_input**n)
    Tref = 25

    tot = (T_input-Tref)/Tref
    deltT = -deltTm+(deltTm+deltTp)*g
    phi = 1/(1+lam*g)
    gamaL=-deltL*tot*phi
    gamaT=deltT *tot
    dL = dL*Q**((T_input-Tref)/10)

    pL = pL*Q**((T_input-Tref)/10)
   
    
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


def calc_metrics(t,y):
    """计算输出指标不变"""
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
    """
    模仿Hopf分岔图输出：输出状态变量的max/min，不输出振幅相位
    返回：factor, x_max(CCA1m最大值), x_min(CCA1m最小值)
    稳态（不振荡）时max≈min
    """
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

        # 舍弃瞬态，取48‑96h稳态
        mask = sol.t >= 48
        y_ss = sol.y[:, mask]
        x = y_ss[4]   # 0号：LHYm/CCA1 mRNA，你可以修改下标

        xmax = np.max(x)
        xmin = np.min(x)
        res.append({"factor":fac, "x_max":xmax, "x_min":xmin})
    return pd.DataFrame(res)



# ====================== 主程序 ======================
p_baseline = np.array([2.7177,0.7601,0.3098,2.2044,5.0094,0.9755,2.7704,2.4447,14.183,5.8215,1.6568,4.8964,0.5554,3.5756,1.1509,3.5684,0.6515,2.6233,0.5447,2.3698,2.0831,0.9166,0.4136,0.7106,1.2734,0.2786,0.1084,0.3042,0.2348,1.2596,0.5022,1.1368,1.5255,0.7036,0.6009,0.9622,5.2985,4.9498,0.8139,0.5654,0.665,0.887,8.8257,0.5281,3.8125,5.2146,3.6288,6.0682,0.6945,0.2286,2.7464,2.5992,0.2938,0.3154,1.473,1.0154,1.6716,1.0154,8.9283,1.9154,4.45745,0.4623,1.0477,234.6663,2.0486,1.4017,4,0.8828  ])

y0_init = np.array([1.52318295e-01,3.39972549e-01,5.95471718e-01,2.15006756e+00,1.35930447e+00,1.09418276e+00,3.15492116e-01,8.46288136e-01,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5])

# y0_init =np.array([0.5, 0.1, 0.85, 0.2,0.5])
I_env, T_env = 450, 37

param_names = [
    "v1","v1L","v2","v2L","v3","v4","v4L","v5","v6","v7","v8",
    "p1","p1L","p2","p3","p4","p5","p6","p7","p8","pL",
    "d1","d2L","d2D","d3L","d3D","d4L","d4D","dL","d5","d6","d7","d8",
    "k1L","k1D","k2","k3","k4","k5","k6","k7","k8",
    "K0","K1","K2","K3","K4","K5","K6","K7","K8","K9","K10","K11","K12","K13","K14","K15","K16","K17",
    "v3L", "deltL", "lam", "LTH", "deltTp", "deltTm", "n","Q"
]

# ========= 修改这里：选择你要做分岔分析的参数下标 =========
scan_idx = 22    # 修改为你需要扫描的参数下标
scan_name = param_names[scan_idx]

df_bif = bifurcation_state_scan(p_baseline, param_idx=scan_idx,
                                y0=y0_init, I_input=I_env, T_input=T_env,
                                factor_min=0.5, factor_max=2, n_points=500)

df_bif.to_csv(f"bifurcation_state_{scan_name}.csv", index=False)

# =================绘图：复刻Hopf图风格====================
plt.rcParams["font.family"] = "Times New Roman"
fig, ax = plt.subplots(figsize=(9,7), dpi=150)

fac = df_bif["factor"]
xmax = df_bif["x_max"]
xmin = df_bif["x_min"]

# 振荡区域：上下两条包络线（对应极限环最大最小）
ax.plot(fac, xmax, color="black", lw=1.2)
ax.plot(fac, xmin, color="black", lw=1.2)

# 稳态区：max与min几乎重合，两条线叠在一起，就是不动点解
ax.axvline(x=1.0, linestyle="--", c="gray", lw=1, label="Best‑fit parameter")

ax.set_xlabel(f"Parameter {scan_name} (fold of best‑fit)", fontsize=11)
# ax.set_ylabel("CCA1 mRNA level", fontsize=11)
ax.set_ylabel("TOC1 mRNA level", fontsize=11)
ax.tick_params(labelsize=9)
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(f"bifurcation_hopf_style_{scan_name}.pdf", bbox_inches="tight")
plt.savefig(f"bifurcation_hopf_style_{scan_name}.tif", bbox_inches="tight", dpi=500)
plt.show()