# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 09:35:37 2021

@author: Huangtin
"""

from scipy.integrate import odeint
from matplotlib.pyplot import MultipleLocator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import matplotlib as mpl
mpl.rcParams["font.sans-serif"] = ["SimHei"]
mpl.rcParams["axes.unicode_minus"] = False

plt.close('all')


df = pd.read_csv("CCA1 expression.csv", header = None, sep = ',')
time = df[0]
mean_CK_LI = df[11]
mean_CK_HI = df[12]
mean_HT_LI = df[13]
mean_HT_HI = df[14]

error_CK_LI = df[16]
error_CK_HI = df[17]
error_HT_LI = df[18]
error_HT_HI = df[19]

df1 = pd.read_csv("TOC1 expression.csv", header = None, sep = ',')
time = df[0]
mean_CK_LI1 = df1[11]
mean_CK_HI1 = df1[12]
mean_HT_LI1 = df1[13]
mean_HT_HI1 = df1[14]

error_CK_LI1 = df1[16]
error_CK_HI1 = df1[17]
error_HT_LI1 = df1[18]
error_HT_HI1 = df1[19]



#%% light input

def lightdark(t):
    if np.mod(t,24)>=12:
        L = 0
        D = 1
    else:
        L = 1
        D = 0  
    return L,D



#%% vector field of minimal model
def fun_CK_LI(y,t):
    L,D = lightdark(t)
    
    
    v1     = 2.7177
    v1L    = 0.7601
    v2     = 0.3098
    v2L    = 2.2044
    v3     = 5.0094
    v4     = 0.9755
    v4L    = 2.7704
    v5     = 2.4447
    v6     = 14.183
    v7     = 5.8215
    v8     = 1.6568
    p1     = 4.8964
    p1L    = 0.5554
    p2     = 3.5756
    p3     = 1.1509
    p4     = 3.5684
    p5     = 0.6515
    p6     = 2.6233
    p7     = 0.5447
    p8     = 2.3698
    pL     = 2.0831
    d1     = 0.9166
    d2L    = 0.4136
    d2D    = 0.7106
    d3L    = 1.2734
    d3D    = 0.2786
    d4L    = 0.1084
    d4D    = 0.3042
    dL     = 0.2348
    d5     = 1.2596
    d6     = 0.5022
    d7     = 1.1368
    d8     = 1.5255
    k1L    = 0.7036
    k1D    = 0.7009
    k2     = 0.9622
    k3     = 10.2985
    k4     = 4.9498
    k5     = 0.8139
    k6     = 0.5654
    k7     = 0.6650
    k8     = 0.8870
    K0     = 8.8257
    K1     = 0.5281
    K2     = 3.8125
    K3     = 5.2146
    K4     = 3.6288
    K5     = 6.0682
    K6     = 0.6945
    K7     = 0.2286
    K8     = 2.7464
    K9     = 2.5992
    K10    = 0.2938
    K11    = 0.3154
    K12    = 1.4730
    K13    = 1.0154
    K14    = 1.6716
    K15    = 1.0154
    K16    = 8.9283
    K17    = 1.9154
    
    
    v3L = 4.45745
    deltL  = 0.4623
    lam    = 1.0477
    LTH    = 300.6663
    deltTp = 2.0486
    deltTm = 1.4017
    n      = 4
    
    I = 150
    g = I**n/(LTH**n+I**n)
    Tref = 25
    T = 25
    
    tot = (T-Tref)/Tref
    deltT = -deltTm+(deltTm+deltTp)*g
    
    phi = 1/(1+lam*g)
    
    gamaL=-deltL*tot*phi
    gamaT=deltT *tot
    
    
    
    
    MCL   = y[0]
    CL    = y[1]
    MP97 = y[2]
    P97  = y[3]
    MP51 = y[4]
    P51  = y[5]
    MEL  = y[6]
    EL   = y[7]   
    P    = y[8]
    
    MLhcb1 = y[9]
    Lhcb1  = y[10]
    MpsbA  = y[11]
    psbA   = y[12]
    MRbcS1 = y[13]
    RbcS1  = y[14]
    MatpA  = y[15]
    atpA   = y[16]
  
    z=np.array([(v1+v1L*P*g+gamaL)/(1+(P97/K1)**2+(P51/K2)**2+(CL/K0)**2)-(k1L*L+k1D*D)*MCL,
                  (p1+p1L*L)*MCL-d1*CL,
                  (v2L*L*P+v2)/(1+(EL/K3)**2+(CL/K4)**2)-k2*MP97,
                  p2*MP97-(d2D*D+d2L*L)*P97,
                  (v3+v3L*P*g+gamaT)/(1+(CL/K6)**2+(P51/K5)**2)-k3*MP51,
                  p3*MP51-(d3D*D+d3L*L)*P51,
                  (v4+v4L*L)/(1+(CL/K7)**2+(P51/K8)**2)*(P97/K9)**2/(1+(P97/K9)**2)-k4*MEL,
                  p4*MEL-(d4D*D+d4L*L)*EL,
                  pL*(1-P)*D-dL*P*L,
                  v5*(CL/K10)**2/(1+(CL/K10)**2)/(1+(P51/K11)**2)-k5*MLhcb1,
                  p5*MLhcb1-d5*Lhcb1,
                  v6/(1+(CL/K12)**2+(P51/K13)**2)-k6*MpsbA,
                  p6*MpsbA-d6*psbA,
                  v7/(1+(CL/K14)**2+(P97/K15)**2)-k7*MRbcS1,
                  p7*MRbcS1-d7*RbcS1,
                  v8/(1+(CL/K16)**2+(P97/K17)**2)-k8*MatpA,
                  p8*MatpA-d8*atpA])
    

    return z


def fun_CK_HI(y,t):
    L,D = lightdark(t)
    
    v1     = 2.7177
    v1L    = 0.7601  
    v2     = 0.3098
    v2L    = 2.2044
    v3     = 5.0094
    v4     = 0.9755
    v4L    = 2.7704
    v5     = 2.4447
    v6     = 14.183
    v7     = 5.8215
    v8     = 1.6568
    p1     = 4.8964
    p1L    = 0.5554
    p2     = 3.5756
    p3     = 1.1509
    p4     = 3.5684
    p5     = 0.6515
    p6     = 2.6233
    p7     = 0.5447
    p8     = 2.3698
    pL     = 2.0831
    d1     = 0.9166
    d2L    = 0.4136
    d2D    = 0.7106
    d3L    = 1.2734
    d3D    = 0.2786
    d4L    = 0.1084
    d4D    = 0.3042
    dL     = 0.2348
    d5     = 1.2596
    d6     = 0.5022
    d7     = 1.1368
    d8     = 1.5255
    k1L    = 0.7036
    k1D    = 0.4009
    k2     = 0.9622
    k3     = 5.2985
    k4     = 4.9498
    k5     = 0.8139
    k6     = 0.5654
    k7     = 0.6650
    k8     = 0.8870
    K0     = 8.8257
    K1     = 0.5281
    K2     = 3.8125
    K3     = 5.2146
    K4     = 3.6288
    K5     = 6.0682
    K6     = 0.6945
    K7     = 0.2286
    K8     = 2.7464
    K9     = 2.5992
    K10    = 0.2938
    K11    = 0.3154
    K12    = 1.4730
    K13    = 1.0154
    K14    = 1.6716
    K15    = 1.0154
    K16    = 8.9283
    K17    = 1.9154
    
    v3L = 4.45745
    deltL  = 0.4623
    lam    = 1.0477
    LTH    = 300.6663
    deltTp = 2.0486
    deltTm = 1.4017   
    n      = 4
    
    I = 450
    g = I**n/(LTH**n+I**n)
    
    Tref = 25
    T = 25
    
    tot = (T-Tref)/Tref
    deltT = -deltTm+(deltTm+deltTp)*g
    
    phi = 1/(1+lam*g)
    
    gamaL=-deltL*tot*phi
    gamaT=deltT *tot
    
    
    
    
    MCL   = y[0]
    CL    = y[1]
    MP97 = y[2]
    P97  = y[3]
    MP51 = y[4]
    P51  = y[5]
    MEL  = y[6]
    EL   = y[7]   
    P    = y[8]
    
    MLhcb1 = y[9]
    Lhcb1  = y[10]
    MpsbA  = y[11]
    psbA   = y[12]
    MRbcS1 = y[13]
    RbcS1  = y[14]
    MatpA  = y[15]
    atpA   = y[16]
  
    z=np.array([(v1+v1L*P*g+gamaL)/(1+(P97/K1)**2+(P51/K2)**2+(CL/K0)**2)-(k1L*L+k1D*D)*MCL,
                  (p1+p1L*L)*MCL-d1*CL,
                  (v2L*L*P+v2)/(1+(EL/K3)**2+(CL/K4)**2)-k2*MP97,
                  p2*MP97-(d2D*D+d2L*L)*P97,
                  (v3+v3L*P*g+gamaT)/(1+(CL/K6)**2+(P51/K5)**2)-k3*MP51,
                  p3*MP51-(d3D*D+d3L*L)*P51,
                  (v4+v4L*L)/(1+(CL/K7)**2+(P51/K8)**2)*(P97/K9)**2/(1+(P97/K9)**2)-k4*MEL,
                  p4*MEL-(d4D*D+d4L*L)*EL,
                  pL*(1-P)*D-dL*P*L,
                  v5*(CL/K10)**2/(1+(CL/K10)**2)/(1+(P51/K11)**2)-k5*MLhcb1,
                  p5*MLhcb1-d5*Lhcb1,
                  v6/(1+(CL/K12)**2+(P51/K13)**2)-k6*MpsbA,
                  p6*MpsbA-d6*psbA,
                  v7/(1+(CL/K14)**2+(P97/K15)**2)-k7*MRbcS1,
                  p7*MRbcS1-d7*RbcS1,
                  v8/(1+(CL/K16)**2+(P97/K17)**2)-k8*MatpA,
                  p8*MatpA-d8*atpA])
    

    return z


def fun_HT_LI(y,t):
    L,D = lightdark(t)

    v1     = 0.6177
    v1L    = 0.7601  
    v2     = 0.3098
    v2L    = 2.2044
    v3     = 1.0094
    v4     = 0.9755
    v4L    = 2.7704
    v5     = 2.4447
    v6     = 14.183
    v7     = 5.8215
    v8     = 1.6568
    p1     = 4.8964
    p1L    = 0.5554
    p2     = 3.5756
    p3     = 1.1509
    p4     = 3.5684
    p5     = 0.6515
    p6     = 2.6233
    p7     = 0.5447
    p8     = 2.3698
    pL     = 2.0831
    d1     = 0.9166
    d2L    = 0.4136
    d2D    = 0.7106
    d3L    = 1.2734
    d3D    = 0.2786
    d4L    = 0.1084
    d4D    = 0.3042
    dL     = 0.2348
    d5     = 1.2596
    d6     = 0.5022
    d7     = 1.1368
    d8     = 1.5255
    k1L    = 0.7036
    k1D    = 0.6009
    k2     = 0.9622
    k3     = 5.2985
    k4     = 4.9498
    k5     = 0.8139
    k6     = 0.5654
    k7     = 0.6650
    k8     = 0.8870
    K0     = 8.8257
    K1     = 0.5281
    K2     = 3.8125
    K3     = 5.2146
    K4     = 3.6288
    K5     = 6.0682
    K6     = 0.6945
    K7     = 0.2286
    K8     = 2.7464
    K9     = 2.5992
    K10    = 0.2938
    K11    = 0.3154
    K12    = 1.4730
    K13    = 1.0154
    K14    = 1.6716
    K15    = 1.0154
    K16    = 8.9283
    K17    = 1.9154
    
    
    v3L = 4.45745
    deltL  = 0.4623
    lam    = 1.0477
    LTH    = 300.6663
    deltTp = 2.0486
    deltTm = 1.4017
    Q      = 0.8828
    n      = 4
    

    I = 150
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
   
    
    
    MCL   = y[0]
    CL    = y[1]
    MP97 = y[2]
    P97  = y[3]
    MP51 = y[4]
    P51  = y[5]
    MEL  = y[6]
    EL   = y[7]   
    P    = y[8]
    
    MLhcb1 = y[9]
    Lhcb1  = y[10]
    MpsbA  = y[11]
    psbA   = y[12]
    MRbcS1 = y[13]
    RbcS1  = y[14]
    MatpA  = y[15]
    atpA   = y[16]
  
    z=np.array([(v1+v1L*P*g+gamaL)/(1+(P97/K1)**2+(P51/K2)**2+(CL/K0)**2)-(k1L*L+k1D*D)*MCL,
                  (p1+p1L*L)*MCL-d1*CL,
                  (v2L*L*P+v2)/(1+(EL/K3)**2+(CL/K4)**2)-k2*MP97,
                  p2*MP97-(d2D*D+d2L*L)*P97,
                  (v3+v3L*P*g+gamaT)/(1+(CL/K6)**2+(P51/K5)**2)-k3*MP51,
                  p3*MP51-(d3D*D+d3L*L)*P51,
                  (v4+v4L*L)/(1+(CL/K7)**2+(P51/K8)**2)*(P97/K9)**2/(1+(P97/K9)**2)-k4*MEL,
                  p4*MEL-(d4D*D+d4L*L)*EL,
                  pL*(1-P)*D-dL*P*L,
                  v5*(CL/K10)**2/(1+(CL/K10)**2)/(1+(P51/K11)**2)-k5*MLhcb1,
                  p5*MLhcb1-d5*Lhcb1,
                  v6/(1+(CL/K12)**2+(P51/K13)**2)-k6*MpsbA,
                  p6*MpsbA-d6*psbA,
                  v7/(1+(CL/K14)**2+(P97/K15)**2)-k7*MRbcS1,
                  p7*MRbcS1-d7*RbcS1,
                  v8/(1+(CL/K16)**2+(P97/K17)**2)-k8*MatpA,
                  p8*MatpA-d8*atpA])
    

    return z


def fun_HT_HI(y,t):
    L,D = lightdark(t)
    
    v1     = 2.7177
    v1L    = 0.7601  
    v2     = 0.3098
    v2L    = 2.2044
    v3     = 5.0094
    v4     = 0.9755
    v4L    = 2.7704
    v5     = 2.4447
    v6     = 14.183
    v7     = 5.8215
    v8     = 1.6568
    p1     = 4.8964
    p1L    = 0.5554
    p2     = 3.5756
    p3     = 1.1509
    p4     = 3.5684
    p5     = 0.6515
    p6     = 2.6233
    p7     = 0.5447
    p8     = 2.3698
    pL     = 2.0831
    d1     = 0.9166
    d2L    = 0.4136
    d2D    = 0.7106
    d3L    = 1.2734
    d3D    = 0.2786
    d4L    = 0.1084
    d4D    = 0.3042
    dL     = 0.2348
    d5     = 1.2596
    d6     = 0.5022
    d7     = 1.1368
    d8     = 1.5255
    k1L    = 0.7036
    k1D    = 0.6009
    k2     = 0.9622
    k3     = 5.2985
    k4     = 4.9498
    k5     = 0.8139
    k6     = 0.5654
    k7     = 0.6650
    k8     = 0.8870
    K0     = 8.8257
    K1     = 0.5281
    K2     = 3.8125
    K3     = 5.2146
    K4     = 3.6288
    K5     = 6.0682
    K6     = 0.6945
    K7     = 0.2286
    K8     = 2.7464
    K9     = 2.5992
    K10    = 0.2938
    K11    = 0.3154
    K12    = 1.4730
    K13    = 1.0154
    K14    = 1.6716
    K15    = 1.0154
    K16    = 8.9283
    K17    = 1.9154
    
    
    v3L = 4.45745
    deltL  = 0.4623
    lam    = 1.0477
    LTH    = 300.6663
    deltTp = 2.0486
    deltTm = 1.4017
    Q      = 0.8828
    n      = 4
    
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
    
    
    MCL   = y[0]
    CL    = y[1]
    MP97 = y[2]
    P97  = y[3]
    MP51 = y[4]
    P51  = y[5]
    MEL  = y[6]
    EL   = y[7]   
    P    = y[8]
    
    MLhcb1 = y[9]
    Lhcb1  = y[10]
    MpsbA  = y[11]
    psbA   = y[12]
    MRbcS1 = y[13]
    RbcS1  = y[14]
    MatpA  = y[15]
    atpA   = y[16]
  
    z=np.array([(v1+v1L*P*g+gamaL)/(1+(P97/K1)**2+(P51/K2)**2+(CL/K0)**2)-(k1L*L+k1D*D)*MCL,
                  (p1+p1L*L)*MCL-d1*CL,
                  (v2L*L*P+v2)/(1+(EL/K3)**2+(CL/K4)**2)-k2*MP97,
                  p2*MP97-(d2D*D+d2L*L)*P97,
                  (v3+v3L*P*g+gamaT)/(1+(CL/K6)**2+(P51/K5)**2)-k3*MP51,
                  p3*MP51-(d3D*D+d3L*L)*P51,
                  (v4+v4L*L)/(1+(CL/K7)**2+(P51/K8)**2)*(P97/K9)**2/(1+(P97/K9)**2)-k4*MEL,
                  p4*MEL-(d4D*D+d4L*L)*EL,
                  pL*(1-P)*D-dL*P*L,
                  v5*(CL/K10)**2/(1+(CL/K10)**2)/(1+(P51/K11)**2)-k5*MLhcb1,
                  p5*MLhcb1-d5*Lhcb1,
                  v6/(1+(CL/K12)**2+(P51/K13)**2)-k6*MpsbA,
                  p6*MpsbA-d6*psbA,
                  v7/(1+(CL/K14)**2+(P97/K15)**2)-k7*MRbcS1,
                  p7*MRbcS1-d7*RbcS1,
                  v8/(1+(CL/K16)**2+(P97/K17)**2)-k8*MatpA,
                  p8*MatpA-d8*atpA])
    

    return z


tspan=np.array([0,240]);
h=0.01
t=np.arange(tspan[0],tspan[1]+h,h)

y0 = np.array([1.52318295e-01,3.39972549e-01,5.95471718e-01,2.15006756e+00,1.35930447e+00,1.09418276e+00,3.15492116e-01,8.46288136e-01,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5])



y_CK_LI = odeint(fun_CK_LI, y0, t)

y_CK_HI = odeint(fun_CK_HI, y0, t)

y_HT_LI = odeint(fun_HT_LI, y0, t)

y_HT_HI = odeint(fun_HT_HI, y0, t)



font = {'family' : 'Times New Roman',
            'weight' : 'normal',
            'size'   : 22,}
font1 = {'family' : 'Times New Roman',
             'weight' : 'normal',
             'size'   : 22,}


fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(121)


x1 = np.linspace(12,24,2)
x2 = np.linspace(36,48,2)
y1 = 0*np.ones(len(x1))
y2 = 3*np.ones(len(x1))
plt.fill_between(x1,y1,y2,facecolor='silver')
plt.fill_between(x2,y1,y2,facecolor='silver')

plt.plot(t-120,y_CK_LI[:,0],color='blue', label='CK-LI(Sim.)', linewidth=2.5)
plt.plot(time, mean_CK_LI, color='blue',linestyle='--',label='CK-LI(Exp.)', linewidth=2.5)


plt.plot(t-120,y_CK_HI[:,0],color='orange', label='CK-HI(Sim.)', linewidth=2.5)
plt.plot(time, mean_CK_HI, color='orange', linestyle='--',label='CK-HI(Exp.)', linewidth=2.5)



plt.plot(t-120,y_HT_LI[:,0],color='green', label='HT-LI(Sim.)', linewidth=2.5)
plt.plot(time, mean_HT_LI, color='green', linestyle='--',label='HT-LI(Exp.)', linewidth=2.5)

plt.plot(t-120,y_HT_HI[:,0],color='red', label='HT-HI(Sim.)', linewidth=2.5)
plt.plot(time, mean_HT_HI, color='red', linestyle='--',label='HT-HI(Exp.)', linewidth=2.5)


plt.xlim(0,48)
plt.ylim(0,3)
plt.xlabel('Time (h)', font1)
plt.ylabel('LHY',fontdict={**font, 'style': 'italic'})
ax1.text(-0.15, 1.03, "A", transform=ax1.transAxes, fontsize=26, fontweight='bold', va='top')

plt.xticks(fontproperties = 'Times New Roman', size = 18)
plt.yticks(fontproperties = 'Times New Roman', size = 18)
plt.legend(prop={'family' : 'Times New Roman', 'size'   : 16})


x_major_locator=MultipleLocator(12)
y_major_locator=MultipleLocator(0.5)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)

ax2 = fig.add_subplot(122)

x1 = np.linspace(12,24,2)
x2 = np.linspace(36,48,2)
y1 = 0*np.ones(len(x1))
y2 = 3*np.ones(len(x1))
plt.fill_between(x1,y1,y2,facecolor='silver')
plt.fill_between(x2,y1,y2,facecolor='silver')


plt.plot(t-120,y_CK_LI[:,4],color='blue', label='CK-LI(Sim.)', linewidth=2.5)
plt.plot(time, mean_CK_LI1, color='blue',linestyle='--', label='CK-LI(Exp.)', linewidth=2.5)



plt.plot(t-120,y_CK_HI[:,4],color='orange', label='CK-HI(Sim.)', linewidth=2.5)
plt.plot(time, mean_CK_HI1, color='orange', linestyle='--',label='CK-HI(Exp.)', linewidth=2.5)



plt.plot(t-120,y_HT_LI[:,4],color='green', label='HT-LI(Sim.)', linewidth=2.5)
plt.plot(time, mean_HT_LI1, color='green', linestyle='--',label='HT-LI(Exp.)', linewidth=2.5)

plt.plot(t-120,y_HT_HI[:,4],color='red', label='HT-HI(Sim.)', linewidth=2.5)
plt.plot(time, mean_HT_HI1, color='red', linestyle='--',label='HT-HI(Exp.)', linewidth=2.5)


plt.xlim(0,48)
plt.ylim(0,3)
plt.xlabel('Time (h)', font1)
plt.ylabel('TOC1',fontdict={**font, 'style': 'italic'})
ax2.text(-0.15, 1.03, "B", transform=ax2.transAxes, fontsize=26, fontweight='bold', va='top')

plt.xticks(fontproperties = 'Times New Roman', size = 18)
plt.yticks(fontproperties = 'Times New Roman', size = 18)
plt.legend(prop={'family' : 'Times New Roman', 'size'   : 16})


x_major_locator=MultipleLocator(12)
y_major_locator=MultipleLocator(0.5)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)

plt.tight_layout(rect=[0,0.05,1,0.98])
plt.savefig("model validation.tif", dpi=500, bbox_inches="tight")
plt.savefig("model validation.pdf", bbox_inches="tight")

plt.savefig("model validation.png", dpi=100, bbox_inches="tight")


plt.show()