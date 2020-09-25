import numpy as np
from matplotlib import pyplot as plt

def get_chebyshev_A(x,order):
    A = np.zeros([len(x),order+1])
    A[:,0]=1
    A[:,1]=x
    for i in range(1,order):
        A[:,i+1]=2*x*A[:,i]-A[:,i-1]
    return A
def get_poly_A(xs,order):
    A=np.zeros([len(x),order+1])
    A[:,0]=1
    for i in range(order):
        A[:,i+1]=x*A[:,i]
    return A

def fit_cheby(x,y,error,order):
    A=get_chebyshev_A(x, order)
    # fit=lin_fit(A,y)
    for o in range(order):
        if o <= 10:
            A_o=A[:,:o]
            fit=lin_fit(A_o,y)
            pred=A_o@fit
            e=np.sqrt(np.mean((pred-y)**2))
            if e <= error:
                return pred, o
        else:
            return print("order not large enough")
            
    
def lin_fit(A,y):
    u,s,v=np.linalg.svd(A,0)
    fitp=v.T@(np.diag(1/s)@(u.T@y))
    return fitp
x= np. linspace(-1,1,100)
y_true = np.log2((x+3)/4)
# sig =1
# y = y_true + sig*np.random.randn(len(x))
y = y_true
# plt.plot(x,y_true)
#choose order
order1 = 20
#fit with truncated cheby
y_pred1=fit_cheby(x,y,1e-6,order1)[0]
order =fit_cheby(x,y,1e-6,order1)[1]
print("truncated Cheby","order=",order)
print('RMS error after fit is ' + repr(np.sqrt(np.mean((y_pred1-y_true)**2))))
print('max error after fit is ' + repr(np.max((y_pred1-y_true)**2)))
# print('RMS error after fit is ' + repr(np.sqrt(np.mean((y_pred1- y_true)**2))))
#fit with chebyshev written by myself
# A2=get_chebyshev_A(x,order)
# u2,s2,v2=np.linalg.svd(A2,0)
# fitp2=v2.T@(np.diag(1/s2)@(u2.T@y))
# y_pred2=A2@fitp2
# y_pred2 = A2@lin_fit(A2,y)
# print("Cheby")
# print('RMS error after fit is ' + repr(np.sqrt(np.mean((y_pred2-y_true)**2))))
# print('max error after fit is ' + repr(np.max((y_pred2-y_true)**2)))
#fit with legendre poly
A3=np.polynomial.legendre.legvander(x,order)
y_pred3 = A3@lin_fit(A3,y)
print("legendre")
print('RMS error after fit is ' + repr(np.sqrt(np.mean((y_pred3-y_true)**2))))
print('max error after fit is ' + repr(np.max((y_pred3-y_true)**2)))
fig1 = plt.figure()
plt.plot(x,y,'*')
plt.plot(x,y_pred1,label="truncated cheby")
# plt.plot(x,y_pred2, label = "written cheby")
plt.plot(x,y_pred3, label = "legendre poly")
fig2 = plt.figure()
plt.scatter(x,y_pred1-y_true,label = "truncated cheby",s=15)
# plt.scatter(x,y_pred2-y_true,label = "written cheby",s=15)
plt.scatter(x,y_pred3-y_true,label = "legendre poly",s=15)
plt.ylim(min((y_pred1-y_true)),max(y_pred1 - y_true))
plt.legend()
plt.show()