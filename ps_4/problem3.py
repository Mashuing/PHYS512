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

def corr(f,g):
    fft1 = np.fft.fft(f)
    fft2 = np.fft.fft(g)
    return np.real(np.fft.ifft(fft1*np.conjugate(fft2)))

x = np.arange(-50,50,0.1)
y = np.exp(-(x-0)**2/8)
kvec  = np.arange(x.size)
yfft = np.fft.fft(y)
dx = 200
dz  =500
# here is the shift gaussian
yfft_200 = yfft*np.exp(-2.0j*np.pi*kvec*dx/x.size)
yfft_500 = yfft*np.exp(-2.0j*np.pi*kvec*dz/x.size)
y_200 = np.fft.ifft(yfft_200)
y_500 = np.fft.ifft(yfft_500)
# did correlation
y_corr200 = corr(y,y_200)
y_corr500 = corr(y,y_500)
fig1, ax1 = plt.subplots()
ax1.plot(x,y,label="before shift")
ax1.plot(x,y_200,label="after shift 200")
ax1.plot(x,y_500,label="after shift 500")
ax1.plot(x,y_corr200,label="correlation_dx=200")
ax1.plot(x,y_corr500,label="correlation_dx=500")
plt.legend()
plt.show()
