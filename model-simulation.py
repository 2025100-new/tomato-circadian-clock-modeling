# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 18:05:03 2026

@author: Ting Huang
"""

import matplotlib
matplotlib.use("Agg")
from scipy.integrate import odeint
from matplotlib.pyplot import MultipleLocator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams["font.sans-serif"] = ["SimHei"]
mpl.rcParams["axes.unicode_minus"] = False

plt.close('all')


#%%  pandas read csv  file
df1 = pd.read_csv("CCA1 expression.csv", header = None, sep = ',')
time = df1[0]
LHY_CK_LI = df1[1]
LHY_CK_HI = df1[2]
LHY_HT_LI = df1[3]
LHY_HT_HI = df1[4]


df2 = pd.read_csv("TOC1 expression.csv", header = None, sep = ',')
time = df2[0]
TOC1_CK_LI = df2[1]
TOC1_CK_HI = df2[2]
TOC1_HT_LI = df2[3]
TOC1_HT_HI = df2[4]

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
def fun_CK_LI(y, t):
    
    L,D = lightdark(t)
   
    vLB    = 0.2645
    vTB    = 0.2466
    vLI    = 2.7672
    vTI    = 0.0155
    v1     = 0.4471
    v2     = 3.9553
    p1     = 4.1742
    p2     = 3.0674
    p3     = 0.0773
    d1     = 0.7877
    d2     = 1.0929
    d3     = 0.3849
    k1     = 1.2059
    k2     = 1.4148
    K1     = 0.4779
    K2     = 1.0556
    K3     = 2.4681
    K4     = 1.3937
    deltL  = 0.5661
    lam    = 1.2104
    LTH    = 300
    deltTp = 1.5761
    deltTm = 0.5966
    Q      = 1.5167
        
    
    I = 150
    n=2.5
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
    
      
    LHYm   = y[0]
    LHY    = y[1]
    TOC1m  = y[2]
    TOC1   = y[3]
    P = y[4]
  
    z=np.array([vLB+vLI*g*P+v1*LHY**2/(K1**2+LHY**2)*TOC1**2/(K2**2+TOC1**2)+gamaL-k1*LHYm,\
                 p1*LHYm-dL*LHY,\
                 vTB+vTI*g*P+v2*K3**2/(K3**2+LHY**2)*TOC1**2/(K4**2+TOC1**2)+gamaT-k2*TOC1m,\
                 p2*TOC1m-dT*TOC1,\
                 p3*(1-P)*D-d3*P*L])
    return z



def fun_CK_HI(y, t):
    
    L,D = lightdark(t)
       
    vLB    = 0.2645
    vTB    = 0.2466
    vLI    = 2.7672
    vTI    = 0.0155
    v1     = 0.4471
    v2     = 3.9553
    p1     = 4.1742
    p2     = 3.0674
    p3     = 0.0773
    d1     = 0.7877
    d2     = 1.0929
    d3     = 0.3849
    k1     = 1.2059
    k2     = 1.4148
    K1     = 0.4779
    K2     = 1.0556
    K3     = 2.4681
    K4     = 1.3937
    deltL  = 0.5661
    lam    = 1.2104
    LTH    = 300
    deltTp = 1.5761
    deltTm = 0.5966
    Q      = 1.5167
    
    
    
    I = 450
    n=2.5
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
    
   

    
    LHYm   = y[0]
    LHY    = y[1]
    TOC1m  = y[2]
    TOC1   = y[3]
    P = y[4]
   

    
  
    z=np.array([vLB+vLI*g*P+v1*LHY**2/(K1**2+LHY**2)*TOC1**2/(K2**2+TOC1**2)+gamaL-k1*LHYm,\
                 p1*LHYm-dL*LHY,\
                 vTB+vTI*g*P+v2*K3**2/(K3**2+LHY**2)*TOC1**2/(K4**2+TOC1**2)+gamaT-k2*TOC1m,\
                 p2*TOC1m-dT*TOC1,\
                 p3*(1-P)*D-d3*P*L])
    return z

def fun_HT_LI(y, t):
    
    L,D = lightdark(t)
    
    
    vLB    = 0.2645
    vTB    = 0.2466
    vLI    = 2.7672
    vTI    = 0.0155
    v1     = 0.4471
    v2     = 3.9553
    p1     = 4.1742
    p2     = 3.0674
    p3     = 0.0773
    d1     = 0.7877
    d2     = 1.0929
    d3     = 0.3849
    k1     = 1.2059
    k2     = 1.4148
    K1     = 0.4779
    K2     = 1.0556
    K3     = 2.4681
    K4     = 1.3937
    deltL  = 0.5661
    lam    = 1.2104
    LTH    = 300
    deltTp = 1.5761
    deltTm = 0.5966
    Q      = 1.5167
    
    
    I = 150
    n=2.5
    g = I**n/(LTH**n+I**n)
    Tref = 25
    T = 37
    
    tot = (T-Tref)/Tref
    deltT = -deltTm+(deltTm+deltTp)*g
    
    phi = 1/(1+lam*g)
    
    gamaL=-deltL*tot*phi
    gamaT=deltT *tot
    
    
    dL = d1*Q**((T-Tref)/10)
    dT = d2*Q**((T-Tref)/10)
    
   
    
    LHYm   = y[0]
    LHY    = y[1]
    TOC1m  = y[2]
    TOC1   = y[3]
    P = y[4]
   

    
  
    z=np.array([vLB+vLI*g*P+v1*LHY**2/(K1**2+LHY**2)*TOC1**2/(K2**2+TOC1**2)+gamaL-k1*LHYm,\
                 p1*LHYm-dL*LHY,\
                 vTB+vTI*g*P+v2*K3**2/(K3**2+LHY**2)*TOC1**2/(K4**2+TOC1**2)+gamaT-k2*TOC1m,\
                 p2*TOC1m-dT*TOC1,\
                 p3*(1-P)*D-d3*P*L])
    return z


def fun_HT_HI(y, t):
    
    L,D = lightdark(t)
    
    

    vLB    = 0.2645
    vTB    = 0.2466
    vLI    = 2.7672
    vTI    = 0.0155
    v1     = 0.4471
    v2     = 3.9553
    p1     = 4.1742
    p2     = 3.0674
    p3     = 0.0773
    d1     = 0.7877
    d2     = 1.0929
    d3     = 0.3849
    k1     = 1.2059
    k2     = 1.4148
    K1     = 0.4779
    K2     = 1.0556
    K3     = 2.4681
    K4     = 1.3937
    deltL  = 0.5661
    lam    = 1.2104
    LTH    = 300
    deltTp = 1.5761
    deltTm = 0.5966
    Q      = 1.5167
    
    
    
    I = 450
    n=2.5
    g = I**n/(LTH**n+I**n)
    Tref = 25
    T = 37
    
    tot = (T-Tref)/Tref
    deltT = -deltTm+(deltTm+deltTp)*g
    
    phi = 1/(1+lam*g)
    
    gamaL=-deltL*tot*phi
    gamaT=deltT *tot
    
    
    dL = d1*Q**((T-Tref)/10)
    dT = d2*Q**((T-Tref)/10)
   
    LHYm   = y[0]
    LHY    = y[1]
    TOC1m  = y[2]
    TOC1   = y[3]
    P = y[4]
   
    

    
  
    z=np.array([vLB+vLI*g*P+v1*LHY**2/(K1**2+LHY**2)*TOC1**2/(K2**2+TOC1**2)+gamaL-k1*LHYm,\
                 p1*LHYm-dL*LHY,\
                 vTB+vTI*g*P+v2*K3**2/(K3**2+LHY**2)*TOC1**2/(K4**2+TOC1**2)+gamaT-k2*TOC1m,\
                 p2*TOC1m-dT*TOC1,\
                 p3*(1-P)*D-d3*P*L])
    return z

tspan=np.array([0,120]);
h=0.01
t=np.arange(tspan[0],tspan[1]+h,h)
#y0 = np.array([1.94479216e-01,0.1,3.19894883e-01,0.1,1.23615127e+00,0.1,0.1,0.1,0.1,0.1,1.63254020e+00,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1])
# y0 = np.array([1.52318295e-01,1.39972549e-01,1.95471718e-01,2.15006756e-01,0.1])
y0_CK_LI = np.array([0.1, 0.1, 0.15, 0.7,0.5])
y0_CK_HI = np.array([0.1, 0.2, 0.65, 0.5,0.5])
y0_HT_LI = np.array([0.5, 0.1, 1.2, 0.7,0.5])
y0_HT_HI = np.array([2.1, 1.2, 0.95, 1.0, 0.4])


y_CK_LI = odeint(fun_CK_LI, y0_CK_LI, t)
y_CK_HI = odeint(fun_CK_HI, y0_CK_HI, t)
y_HT_LI = odeint(fun_HT_LI, y0_HT_LI, t)
y_HT_HI = odeint(fun_HT_HI, y0_HT_HI, t)


font = {'family' : 'Times New Roman',
            'weight' : 'normal',
            'size'   : 20,}
font1 = {'family' : 'Times New Roman',
             'weight' : 'normal',
             'size'   : 14,}

plt.rcParams['font.family'] = 'Times New Roman'
plt.figure(figsize=(16, 9))

plt.subplot(121)

x1 = np.linspace(12,24,2)
x2 = np.linspace(36,48,2)
y1 = 0*np.ones(len(x1))
y2 = 3*np.ones(len(x1))

plt.fill_between(x1,y1,y2,facecolor='silver')
plt.fill_between(x2,y1,y2,facecolor='silver')


plt.plot(t-72,y_CK_LI[:,0]*3-1.6,color='#1f77b4',lw=2.4,label='CK-LI (Sim.)')
plt.scatter(time,LHY_CK_LI,marker='o',color='grey')
plt.plot(time,LHY_CK_LI,linestyle = '-.',color='#1f77b4',lw=2.0,label='CK-LI (Exp.)')



plt.plot(t-72,y_CK_HI[:,0]*2.4-1.2,color='#ff7f0e',lw=2.4,label='CK-HI (Sim.)')
plt.scatter(time,LHY_CK_HI,marker='o',color='grey')
plt.plot(time,LHY_CK_HI,linestyle = '-.',color='#ff7f0e',lw=2.0,label='CK-HI (Exp.)')


plt.plot(t,y_HT_LI[:,0]/10,color='#2ca02c',lw=2.4,label='HT-LI (Sim.)')
plt.scatter(time,LHY_HT_LI,marker='o',color='grey')
plt.plot(time,LHY_HT_LI,linestyle = '-.',color='#2ca02c',lw=2.0,label='HT-LI (Exp.)')

plt.plot(t,y_HT_HI[:,0]-0.3,color='#d62728',lw=2.4,label='HT-HI (Sim.)')
plt.scatter(time,LHY_HT_HI,marker='o',color='grey')
plt.plot(time,LHY_HT_HI,linestyle = '-.',color='#d62728',lw=2.0,label='HT-HI (Exp.)')

plt.title('RMSE=0.4806', fontsize=20)

plt.text(-0.1, 1.03, "A", transform=plt.gca().transAxes, fontsize=26, fontweight='bold', va='top')
plt.xlim(0,48)
plt.ylim(0,3)
plt.legend(prop={'family' : 'Times New Roman', 'size'   : 14})
plt.xticks(fontproperties = 'Times New Roman', size = 16)
plt.yticks(fontproperties = 'Times New Roman', size = 16)
plt.xlabel('Time(h)',font)
plt.ylabel('LHY',fontdict={**font, 'style': 'italic'})
plt.tight_layout()




x_major_locator=MultipleLocator(12)
y_major_locator=MultipleLocator(0.5)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)





plt.subplot(122)
x1 = np.linspace(12,24,2)
x2 = np.linspace(36,48,2)
y1 = 0*np.ones(len(x1))
y2 = 3*np.ones(len(x1))

plt.fill_between(x1,y1,y2,facecolor='silver')
plt.fill_between(x2,y1,y2,facecolor='silver')

plt.plot(t-72,y_CK_LI[:,2]-0.7,color='#1f77b4',lw=2.4, label='CK-LI (Sim.)')
plt.scatter(time,TOC1_CK_LI,marker='o',color='grey')
plt.plot(time,TOC1_CK_LI,linestyle = '-.',color='#1f77b4',lw=2.0, label='CK-LI (Exp.)')


plt.plot(t-72,y_CK_HI[:,2]*1.1-0.2,color='#ff7f0e',lw=2.4,label='CK-HI (Sim.)')
plt.scatter(time,TOC1_CK_HI,marker='o',color='grey')
plt.plot(time,TOC1_CK_HI,linestyle = '-.',color='#ff7f0e',lw=2.0,label='CK-HI (Exp.)')


plt.plot(t,y_HT_LI[:,2]/20,color='#2ca02c',lw=2.4,label='HT-LI (Sim.)')
plt.scatter(time,TOC1_HT_LI,marker='o',color='grey')
plt.plot(time,TOC1_HT_LI,linestyle = '-.',color='#2ca02c',lw=2.0,label='HT-LI (Exp.)')

plt.plot(t,y_HT_HI[:,2]-0.9,color='#d62728',lw=2.4,label='HT-HI (Sim.)')
plt.scatter(time,TOC1_HT_HI,marker='o',color='grey')
plt.plot(time,TOC1_HT_HI,linestyle = '-.',color='#d62728',lw=2.0,label='HT-HI (Exp.)')
plt.text(-0.1, 1.03, "B", transform=plt.gca().transAxes, fontsize=26, fontweight='bold', va='top')

plt.title('RMSE=0.4021', fontsize=20)

plt.xlim(0,48)
plt.ylim(0,3)
plt.legend(prop={'family' : 'Times New Roman', 'size'   : 14})
plt.xticks(fontproperties = 'Times New Roman', size = 16)
plt.yticks(fontproperties = 'Times New Roman', size = 16)
plt.xlabel('Time(h)',font)
plt.ylabel('TOC1',fontdict={**font, 'style': 'italic'})
plt.tight_layout()



x_major_locator=MultipleLocator(12)
y_major_locator=MultipleLocator(0.5)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)


plt.tight_layout(rect=[0.07, 0.08, 1.1, 0.95])
plt.savefig("Figure 6.tif", dpi=500, bbox_inches="tight")
plt.savefig("Figure 6.pdf", bbox_inches="tight")
# plt.show()