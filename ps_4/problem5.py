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
# c
def my_dft(k,y):
    N = len(y)
    x=np.arange(N)
    kvec = np.fft.fftfreq(N,1/N) 
    F = np.zeros(N)
    i=0
    for kveci in kvec:
        F_i = np.sum(np.exp(-2*np.pi*1j*(kveci-k)*x/N)/2j-np.exp(-2*np.pi*1j*(kveci+k)*x/N)/2j)
        F[i] = abs(F_i)
        i+=1
    return kvec, F
N=1024
x=np.arange(N)
k1 =15.2
k2=15.2
y1=np.sin(2*np.pi*x*k1/N)
y2=np.sin(2*np.pi*x*k2/N)
y1ft = np.fft.fft(y1)
y2ft = my_dft(k2,y2)[1]
y_true = np.zeros(len(y2ft))
y_true[15]=512.0
y_true[1011]=512.0
fig1, ax1  =plt.subplots()
plt.plot(abs(y1ft),'.',label = "numpy fft")
plt.plot(abs(y2ft),'.',label = "Written dft")
plt.plot(abs(y_true),'.',label = "delta function")
plt.legend()
plt.show()
error = np.std(abs(y1ft)-abs(y2ft))
error_truth = np.std(abs(y1ft)-y_true)
print("Error between written DFT and numpy FFT=",error)
print("Error between DFT and delta function", error_truth)
# d
N=1024
x=np.arange(N)
xx=np.linspace(0,1,N)*2*np.pi
win=0.5-0.5*np.cos(xx)
k1 =15.2
k2=15.2
y1=np.sin(2*np.pi*x*k1/N)
y2=np.sin(2*np.pi*x*k2/N)*win
y1ft = np.fft.fft(y1)
y2ft = np.fft.fft(y2)
y_true = np.zeros(len(y2ft))
y_true[15]=512.0
y_true[1011]=512.0
plt.plot(abs(y1ft),'.',label="unwindowed DFT")
plt.plot(abs(y2ft),'.',label="windowed DFT")
plt.plot(abs(y_true),'.',label="Delta function")
plt.legend()
plt.show()
error_truth1 = np.std(abs(y1ft)-y_true)
error_truth2 = np.std(abs(y2ft)-y_true)
print("Error without window",error_truth1)
print("Error with window", error_truth2)

