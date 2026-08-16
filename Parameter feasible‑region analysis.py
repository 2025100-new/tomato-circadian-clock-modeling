# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 17:25:50 2026

@author: FHC-Desktop
"""

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

# ===================== 1. 光暗周期函数 =====================
def lightdark(t):
    if np.mod(t,24)>=12:
        L = 0
        D = 1
    else:
        L = 1
        D = 0
    return L,D

# --------------------------
# 你的ODE函数，复用你原有
def ode_system_env(t, y, p, I_input, T_input):
    LHYm, LHY, TOC1m, TOC1, P = y
    vLB, vTB, vLI, vTI, v1, v2, p1, p2, p3, d1, d2, d3, k1,  k2, K1, K2, K3, K4, deltL, lam, LTH, deltTp, deltTm, Q, n = p
    L,D = lightdark(t)
    I_input = 450
    g = I_input**n/(LTH**n+I_input**n)
    Tref = 25
    T_input = 37
    
    tot = (T_input-Tref)/Tref
    
    deltT = -deltTm+(deltTm+deltTp)*g
    
    phi = 1/(1+lam*g)
    
    gamaL=-deltL*tot*phi
    gamaT=deltT *tot
    
    dL = d1*Q**((T_input-Tref)/10)
    dT = d2*Q**((T_input-Tref)/10)
    
    dLHYmdt = vLB+vLI*g*P+v1*LHY**2/(K1**2+LHY**2)*TOC1**2/(K2**2+TOC1**2)+gamaL-k1*LHYm
    dLHYdt = p1*LHYm-dL*LHY
    dTOC1mdt = vTB+vTI*g*P+v2*K3**2/(K3**2+LHY**2)*TOC1**2/(K4**2+TOC1**2)+gamaT-k2*TOC1m
    dTOC1dt = p2*TOC1m-dT*TOC1
    dPdt = p3*(1-P)*D - d3*P*L 
    return [dLHYmdt, dLHYdt, dTOC1mdt, dTOC1dt, dPdt]

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


# =========约束条件（你可以修改）=========
PHASE_CCA1_MIN = 22
PHASE_CCA1_MAX = 26
PHASE_TOC1_MIN = 10
PHASE_TOC1_MAX = 16

def check_constraint(t,y):
    """判断当前解是否满足相位约束，并且存在振荡"""
    met = calc_metrics(t,y)
    cca1_phase = met["cca1_phase"]
    toc1_phase = met["toc1_phase"]
    amp_cca = met["cca1_amp"]
    amp_toc = met["toc1_amp"]
    # 必须同时：相位落在区间，振幅>0.01（保证存在振荡，不是稳态）
    ok = (PHASE_CCA1_MIN <= cca1_phase <= PHASE_CCA1_MAX) and \
         (PHASE_TOC1_MIN <= toc1_phase <= PHASE_TOC1_MAX) and \
         (amp_cca>0.01) and (amp_toc>0.01)
    return ok, met


def scan_single_param(p_base, idx_param, y0, I_input, T_input,
                      search_range=(0.01, 3.0), num_points=200):
    """
    一维扫描单个参数
    idx_param：待扫描参数下标
    search_range：相对于基准值倍数，0.01~3倍基准
    return: p_low, p_high:满足约束的参数上下限（绝对值）
    """
    p0 = p_base[idx_param]
    test_factors = np.linspace(search_range[0], search_range[1], num_points)
    valid_p = []
    for fac in test_factors:
        p_test = np.copy(p_base)
        p_test[idx_param] = p0 * fac
        sol = solve_ivp(ode_system_env, t_span=(0,96), y0=y0,
                        args=(p_test,I_input,T_input), t_eval=np.arange(0,96,0.5), method="RK45")
        if not sol.success:
            continue
        ok,_ = check_constraint(sol.t, sol.y)
        if ok:
            valid_p.append(p0*fac)
    if len(valid_p)==0:
        return np.nan, np.nan
    return min(valid_p), max(valid_p)


# =================主程序====================
p_baseline = np.array([0.2645,0.2466,2.7672,0.0255,0.4471,3.9553,4.1742,3.0674,0.0773,0.7877,1.0929,0.3849,1.2059,1.4148,0.4779,1.0556,2.4681,1.3937,0.5661,1.2104, 300, 1.5761,0.5966,1.5167,2.5])
y0_init = [1.0,0.2,0.8,0.3,0.5]
# 选一个环境做扫描，CK‑LI
I_env, T_env = 450, 25

param_names = [
    "vLB","vTB","vLI","vTI","v1","v2","p1","p2","p3",
    "d1","d2","d3","k1","k2","K1","K2","K3","K4",
    "deltL","lam","LTH","deltTp","deltTm","Q","n"
]

res_list = []
for i,par_name in enumerate(param_names):
    print(f"scanning parameter: {par_name}")
    p_low, p_high = scan_single_param(p_baseline, idx_param=i, y0=y0_init,
                                      I_input=I_env, T_input=T_env)
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
df_scan.to_csv("parameter_feasible_range_phase_constrain.csv",index=False)
print(df_scan.round(3))

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Times New Roman"

fig, ax = plt.subplots(figsize=(10, 12), dpi=120)
df_valid = df_scan.dropna().copy()
y_pos = np.arange(len(df_valid))
base_fold = 1.0

# 可行区间上下限
fold_low  = df_valid["fold_low"].values
fold_high = df_valid["fold_high"].values

# 绘制横向填充色块（可行域）
bar_height = 0.35
for i, y in enumerate(y_pos):
    x0 = fold_low[i]
    x1 = fold_high[i]
    ax.fill_betweenx([y-bar_height/2, y+bar_height/2], x0, x1, 
                     color="#6080cc", alpha=0.55)

# 最优值圆点
ax.scatter(np.full(len(df_valid), base_fold), y_pos, 
           color="#204099", zorder=10, s=35)

ax.axvline(x=1, linestyle="--", color="gray", label="Best‑fit value (1.0 fold)")

ax.set_yticks(y_pos)
ax.set_yticklabels(df_valid["param"], fontsize=11)
ax.set_xlabel("Relative parameter value (fold of best‑fit)", fontsize=14)
ax.set_ylabel("Model parameter", fontsize=14)
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig("parameter_feasible_region.pdf", bbox_inches="tight")
plt.savefig("parameter_feasible_region.png", bbox_inches="tight", dpi=300)
plt.show()
