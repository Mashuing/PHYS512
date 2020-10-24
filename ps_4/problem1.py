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

def shift(f,dx):
    kvec  = np.arange(f.size)
    yfft = np.fft.fft(f)
    dx = 500
    yfft_new = yfft*np.exp(-2.0j*np.pi*kvec*dx/x.size)
    y_new = np.fft.ifft(yfft_new)
    return y_new

x = np.arange(-50,50,0.1)
y = np.exp(-(x-0)**2/8)
y_new = shift(y,500)
fig1, ax1 = plt.subplots()
ax1.plot(x,y,label="before shift")
ax1.plot(x,abs(y_new),label="after shift")
plt.legend()
plt.show()

