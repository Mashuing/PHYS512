# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 23:55:34 2020

@author: Jiaxing_Home
"""

import numpy as np
import matplotlib.pyplot as plt
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
plt.rcParams['axes.prop_cycle'    ] = plt.cycler('color',['lightseagreen', 'indigo', 'dodgerblue', 'sandybrown', 'brown', 'coral', 'pink' ])
def dofft(y):
    f = np.fft.rfft(y)
    x = np.arange(f.size)
    return x[1:], f[1:]
chain1 = np.loadtxt("chain_noprior.txt")
chivec1 = np.loadtxt("chivec_noprior.txt")
chain2 = np.loadtxt("chain_withprior.txt")
chivec2 = np.loadtxt("chivec_withprior.txt")
n = np.linspace(1,10000,10000)
scale = [1e1,1e-2,1e-1,1e-1,1e-9,1e-1]
fig1,ax1 = plt.subplots(2,1)
chain1_rescale = chain1/scale
chain2_rescale = chain2/scale
mean1 = np.mean(chain1,axis=0)
sigma1 = np.std(chain1,axis=0)
mean2 = np.mean(chain2,axis=0)
sigma2 = np.std(chain2,axis=0)
print("No prior tau, pararmeters = "+str(mean1)+ "Error in parameters= "+str(sigma1))
for i in range(6):
    ax1[0].plot(n,chain1_rescale[:,i],label = "parameter number = "+str(i))
    # ax1[1].loglog(n,chain1_rescale[:,i],label = "parameter number = "+str(i))
    fft_x, fft_y = dofft(chain1_rescale[:,i])
    ax1[1].plot(fft_x,fft_y,label = "parameter number = "+str(i))
plt.legend()
plt.show()
# With prior tau
fig2,ax2 = plt.subplots(2,1)
print("With prior tau, pararmeters = "+str(mean2)+ "Error in parameters= "+str(sigma2))
for i in range(6):
    ax2[0].plot(n,chain2_rescale[:,i],label = "parameter number = "+str(i))
    # ax2[1].loglog(n,chain2_rescale[:,i],label = "parameter number = "+str(i))
    fft_x, fft_y = dofft(chain2_rescale[:,i])
    ax2[1].plot(fft_x,fft_y,label = "parameter number = "+str(i))
plt.legend()
plt.show()    
    
# Important Sampling
# get weight vector
wtvec=np.exp(-0.5*((chain1[:,3]-0.0544)/0.0073)**2)
chain1_scat=chain1.copy()
means=np.zeros(chain1.shape[1])
chain1_errs=np.zeros(chain1.shape[1])
for i in range(chain1.shape[1]):
    # weight the parameters
    means[i]=np.sum(wtvec*chain1[:,i])/np.sum(wtvec)
    #subtract the mean from the warm chain so we can calculate the
    #standard deviation
    chain1_scat[:,i]=chain1_scat[:,i]-means[i]
    chain1_errs[i]=np.sqrt(np.sum(chain1_scat[:,i]**2*wtvec)/np.sum(wtvec))

chain1_scat_=chain1_scat/scale
mean3 = np.mean(chain1_scat,axis=0)
sigma3 = np.std(chain1_scat,axis=0)
print("With important sample by tau, pararmeters = "+str(means)+ "Error in parameters= "+str(chain1_errs))
