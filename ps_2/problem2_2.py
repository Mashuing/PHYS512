import numpy as np
from matplotlib import pyplot as plt

def get_chebyshev_A(x,order):
    A = np.zeros([len(x),order+1])
    A[:,0]=1
    A[:,1]=x
    for i in range(1,order):
        A[:,i+1]=2*x*A[:,i]-A[:,i-1]
    return A
    

x= np. linspace(0.5,1,5001)
y_true = np.log(x)/np.log(2)
sig = 0.1
# y = y_true + sig*np.random.randn(len(x))
y = y_true

plt.ion()
plt.clf()
# plt.plot(x,y_true)
plt.plot(x,y,'*')
#choose order
order = 100
#fit with numpy chebyshev
A=np.polynomial.chebyshev.chebvander(x,order)
u,s,v=np.linalg.svd(A,0)
fitp=v.T@(np.diag(1/s)@(u.T@y))
y_pred_cheb=A@fitp
print('RMS error after fit is ' + repr(np.sqrt(np.mean((y_pred_cheb-y_true)**2))))
#fit with chebyshev written by myself
A2=get_chebyshev_A(x,order)
u2,s2,v2=np.linalg.svd(A2,0)
fitp2=v2.T@(np.diag(1/s2)@(u2.T@y))
y_pred2=A2@fitp2
print('RMS error after fit is ' + repr(np.sqrt(np.mean((y_pred2-y_true)**2))))
print('max error after fit is ' + repr(np.max((y_pred2-y_true)**2)))
#fit with legendre
A3=np.polynomial.legendre.legvander(x,order)
u3,s3,v3=np.linalg.svd(A3,0)
fitp3=v3.T@(np.diag(1/s3)@(u3.T@y))
y_pred3=A3@fitp3
print('RMS error after fit is ' + repr(np.sqrt(np.mean((y_pred3-y_true)**2))))
print('max error after fit is ' + repr(np.max((y_pred3-y_true)**2)))
plt.plot(x,y_pred_cheb,label="numpy cheby")
plt.plot(x,y_pred2, label = "written cheby")
plt.plot(x,y_pred3, label = "numpy legendre")
plt.legend()
plt.show()