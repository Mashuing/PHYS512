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
# exponential with ratio of uniforms
n = 1000000
u = np.random.rand(n)
v = np.random.rand(n)*0.75
rat = v/u
accept = u<np.sqrt(np.exp(-1*rat))
myexp = rat[accept]
fig1, ax1 = plt.subplots()
ee=myexp[myexp<10]
c,d = np.histogram(ee,100)
dd = 0.5*(d[:-1]+d[1:])
pred = np.exp(-1*dd)
pred = pred/pred[0]*c[0]
ax1.bar(dd,c,width=1/dd.max(),label="histogram of exponential")
ax1.plot(dd,pred,"r",label="exponential")
plt.legend()
print("Accept fraction is ", np.mean(accept))