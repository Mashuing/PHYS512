import numpy as np
from scipy import integrate
import time
from matplotlib import pyplot as plt

def fun(x,y,half_life=[4468e9*365*3600,2410*3600,670*60,245500*3600*365,75380*365*3600,1600*365*3600,38235*3600,310*60,268*60,199*60,1643e-3,223*365*3600,5015*365*3600,138376*3600]):
    dydx=np.zeros(len(half_life)+1)
    dydx[0]=-y[0]/half_life[0]
    for i in range(1,len(half_life)):
        dydx[i]=y[i-1]/half_life[i-1]-y[i]/half_life[i]
    dydx[len(half_life)]=y[len(half_life)-1]/half_life[len(half_life)-1]
    return dydx
half_life=[4468e9*365*3600,2410*3600,670*60,245500*3600*365,75380*365*3600,1600*365*3600,38235*3600,310*60,268*60,199*60,1643e-3,223*365*3600,5015*365*3600,138376*3600]
y0=np.zeros(15)
y0[0]=1
x0=0
x1=5e19
t1=time.time()
ans_stiff=integrate.solve_ivp(fun,[x0,x1],y0,method='Radau')
t2=time.time()
print('took ',ans_stiff.nfev,' evaluations and ',t2-t1,' seconds to solve implicitly')
t= ans_stiff.t
Pb206 = ans_stiff.y[14]
U238 = ans_stiff.y[0]
U234 = ans_stiff.y[3]
T230 = ans_stiff.y[4]
fig1 = plt.figure()
# plt.plot(t,U238,label="U238")
plt.plot(t,Pb206/U238,label="Pb206 to U238")
plt.xlabel("time/s")
plt.legend()
plt.show()
fig2 = plt.figure()
plt.plot(t,U234/T230, label="U234/Th230")
# plt.xlim(1e8,0.5e12)
plt.xlabel("time/s")
plt.ylim(0,200)
plt.legend()
plt.show()
