import matplotlib.pyplot as plt
import numpy as np
import camb
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
# fix tau, change the other parameters
pars=np.asarray([65,0.02,0.1,2e-9,0.96])
# pars_guess = np.asarray([60,0.01,0.08,0.05,3e-9,1.20])
wmap=np.loadtxt('wmap_tt_spectrum_9yr_v5.txt')
x = wmap[:,0]
y=  wmap[:,1]
noise  = wmap[:,2]
def get_spectrum(pars,tau=0.05,lmax=2000):
    # print('pars are ',pars)
    H0=pars[0]
    ombh2=pars[1]
    omch2=pars[2]
    tau=tau
    As=pars[3]
    ns=pars[4]
    pars=camb.CAMBparams()
    pars.set_cosmology(H0=H0,ombh2=ombh2,omch2=omch2,mnu=0.06,omk=0,tau=tau)
    pars.InitPower.set_params(As=As,ns=ns,r=0)
    pars.set_for_lmax(lmax,lens_potential_accuracy=0)
    results=camb.get_results(pars)
    powers=results.get_cmb_power_spectra(pars,CMB_unit='muK')
    cmb=powers['total']
    pred = cmb[2:len(wmap[:,0])+2,0]
    return pred

def num_deriv(fun,x,pars,dpar):
    #calculate numerical derivatives of 
    #a function for use in e.g. Newton's method or LM
    derivs=np.zeros([len(x),len(pars)])
    for i in range(len(pars)):
        pars2=pars.copy()
        pars2[i]=pars2[i]+dpar[i]
        f_right=fun(pars2)
        pars2[i]=pars[i]-dpar[i]
        f_left=fun(pars2)
        derivs[:,i]=(f_right-f_left)/(2*dpar[i])
    return derivs
Ninv=np.eye(len(x))/noise**2
dpar=pars/10.0
pars = pars
tau = 0.05
dtau =tau/10.
tol = 1e-6
chisq=np.sum(((y-get_spectrum(pars))/noise)**2)+2*tol
for i in range(10):
    model=get_spectrum(pars)
    t_chisq = np.sum(((y-model)/noise)**2)
    if 0 < chisq - t_chisq < tol:
        print ("get the good result")
        break
    chisq = t_chisq
    derivs=num_deriv(get_spectrum,x,pars,dpar)
    resid=y-model
    lhs=derivs.T@Ninv@derivs
    rhs=derivs.T@Ninv@resid
    lhs_inv=np.linalg.inv(lhs)
    step=lhs_inv@rhs
    pars=pars+step
    print(pars, chisq)
# since we have a curvature estimate from Newton's method, we can
# guess our chain sampling using that

#  without consider tau
par_sigs=np.sqrt(np.diag(lhs_inv))
par_errs=np.sqrt(np.diag(np.linalg.inv(lhs)))
print('final parameters are ',pars,' with errors ',par_errs)
pars_best = pars
pars = np.insert(pars,3,tau)
f_r = get_spectrum(pars_best,tau+dtau)
f_l = get_spectrum(pars_best,tau-dtau)
derivs_tau = (f_r-f_l)/(2*dtau)
derivs = np.insert(derivs,3,derivs_tau,1)
lhs=derivs.T@Ninv@derivs
rhs=derivs.T@Ninv@resid
lhs_inv=np.linalg.inv(lhs)
#  with consider tau
par_sigs=np.sqrt(np.diag(lhs_inv))
par_errs=np.sqrt(np.diag(np.linalg.inv(lhs)))
print('final parameters are ',pars,' with errors ',par_errs)
fig1, ax1 = plt.subplots()
# ax1.errorbar(wmap[:,0],wmap[:,1],wmap[:,2],fmt='*')
ax1.plot(wmap[:,0],wmap[:,1],'.')

cmb=get_spectrum(pars_best)
sig  = wmap[:,2]
pred = cmb
chisq = np.sum(((wmap[:,1]-pred)/sig)**2)
ax1.plot(pred)
plt.legend()
plt.show()
print("chisquare = ", chisq)




