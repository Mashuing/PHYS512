import numpy as np
import matplotlib.pyplot as plt
import time
''' The following is just cosmectic change'''
plt.rcParams['mathtext.fontset'   ] = 'cm' 
plt.rcParams['font.sans-serif'    ] = 'Arial'
plt.rcParams['figure.figsize'     ] = 10, 8
plt.rcParams['font.size'          ] = 19
plt.rcParams['lines.linewidth'    ] = 2
plt.rcParams['xtick.major.width'  ] = 2
plt.rcParams['ytick.major.width'  ] = 2
plt.rcParams['xtick.major.pad'    ] = 4
plt.rcParams['ytick.major.pad'    ] = 4
plt.rcParams['xtick.major.size'   ] = 10
plt.rcParams['ytick.major.size'   ] = 10
plt.rcParams['axes.linewidth'     ] = 2
plt.rcParams['patch.linewidth'    ] = 0
plt.rcParams['legend.fontsize'    ] = 15
plt.rcParams['xtick.direction'    ] = 'in'
plt.rcParams['ytick.direction'    ] = 'in'
plt.rcParams['ytick.right'        ] = True
plt.rcParams['xtick.top'          ] = True
plt.rcParams['xtick.minor.width'  ] = 1
plt.rcParams['xtick.minor.size'   ] = 4
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['ytick.minor.visible'] = True
plt.rcParams['ytick.minor.width'  ] = 1
plt.rcParams['ytick.minor.size'   ] = 4
plt.rcParams['axes.labelpad'      ] = 0
def random_power(a=-3):
    r=np.random.rand(100000)
    alpha = a
    x = (1-r)**(1/(1+alpha))
    return x
def expfrompower(x):
    accept_prob=(1/1.355)*np.exp(-x)/(x**-3)
    assert(np.max(accept_prob<=1))
    accept=np.random.rand(len(accept_prob))<accept_prob
    return x[accept]
def exp(tau):
    y=np.random.rand(100000)
    t = -tau*np.log(1-y)
    return t
# Generate random power law
fig1, ax1 = plt.subplots()
x = random_power(a=-3)
xx=x[x<10]
a,b = np.histogram(xx,100)
bb = 0.5*(b[:-1]+b[1:])
ax1.bar(bb,1.355*a/np.max(a),width=0.9/bb.max(),label="histogram of power laws")
pred = bb**-3
pred=pred/pred[0]*a[0]
ax1.plot(bb,1.355*pred/np.max(pred),"r",label="power laws")
ax1.plot(bb,np.exp(-1*bb),"b",label="exponential")
plt.legend()
plt.show()
fig2, ax2 = plt.subplots()
# From random power law to exponential
e = expfrompower(x)
print('accept fraction is', len(e)/len(x))
ee=e[e<10]
c,d = np.histogram(ee,100)
dd = 0.5*(d[:-1]+d[1:])
pred = np.exp(-1*dd)
pred = pred/pred[0]*c[0]
ax2.bar(dd,c,width=0.95/dd.max(),label="histogram of exponential")
ax2.plot(dd,pred,"r",label="exponential")
plt.legend()
# Efficeny test
fig3, ax3 = plt.subplots()
t1=time.time()
x_2 = random_power(a=-3)
e_2 = expfrompower(x_2)
t2 = time.time()
e_3 = exp(tau=1)
t3 = time.time()
print("Time from rejection method:", t2-t1)
print("Time from transformation method:", t3-t2)
