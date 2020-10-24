import numpy as np
import matplotlib.pyplot as plt

''' The following is just cosmectic change'''
plt.rcParams['mathtext.fontset'   ] = 'cm' 
plt.rcParams['font.sans-serif'    ] = 'Arial'
plt.rcParams['figure.figsize'     ] = 10, 8
plt.rcParams['font.size'          ] = 19
plt.rcParams['lines.linewidth'    ] = 4
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

def conv(f,g,win):
    f = f*win
    g = g*win
    fft1 = np.fft.fft(f)
    fft2 = np.fft.fft(g)
    return np.real(np.fft.ifft(fft1*fft2))
N=1024
x=np.arange(N)
k_1=13.5
k_2=15.5
y=np.cos(2*np.pi*x*k_1/N)
z=np.cos(2*np.pi*x*k_2/N)
win_one = np.ones(len(y))
win = np.ones(len(y))
win[-3:]=0
win[:3]=0
con_1 = conv(y,y,win_one)
con_2 = conv(y,y,win)
fig1, ax1 = plt.subplots()
ax1.plot(con_1,label="no win")
ax1.plot(con_2,label="win")
plt.legend()
plt.show()