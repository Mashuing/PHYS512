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
    fit=lin_fit(A,y)
    for o in range(1, order):
        pred=A[:,:o]@fit[:o]
        e=np.abs(pred[-1])
        if e <= error:
            return pred, o-1,e
    return "Maximum order, not correct"
               
def lin_fit(A,y):
    u,s,v=np.linalg.svd(A,0)
    fitp=v.T@(np.diag(1/s)@(u.T@y))
    return fitp
x= np. linspace(-1,1,201)
x_t = np.linspace(0.5,1,len(x))
y_true = np.log2((x+3)/4)
# sig =1
# y = y_true + sig*np.random.randn(len(x))
y = y_true
# plt.plot(x,y_true)
#choose maximum order for cheby
order1 = 20
#fit with truncated cheby
y_pred1=fit_cheby(x,y,1e-6,order1)[0]
order =fit_cheby(x,y,1e-6,order1)[1]
error = fit_cheby(x,y,1e-6,order1)[2]
print("truncated Cheby","order=",order,"Error at order",error)
print('RMS error after fit is ' + repr(np.sqrt(np.mean((y_pred1-y_true)**2))))
print('max error after fit is ' + repr(np.sqrt(np.max((y_pred1-y_true)**2))))

#fit with Regular poly
A3=get_poly_A(x,order)
y_pred3 = A3@lin_fit(A3,y)
print("legendre")
print('RMS error after fit is ' + repr(np.sqrt(np.mean((y_pred3-y_true)**2))))
print('max error after fit is ' + repr(np.sqrt(np.max((y_pred3-y_true)**2))))
fig1 = plt.figure()
plt.scatter(x_t,y,label="True Value",s = 100,marker="+")
plt.title("Cheby and poly fit")
plt.plot(x_t,y_pred1,label="truncated cheby",color = "Green")
# plt.plot(x,y_pred2, label = "written cheby")
plt.plot(x_t,y_pred3, label = "Regular poly",color="red")
plt.legend()
plt.show()
fig2 = plt.figure()
y_pred1_d = y_pred1-y_true
y_pred3_d = y_pred3-y_true
plt.scatter(x_t,y_pred1_d,label = "truncated cheby",s=15)
plt.scatter(x_t,y_pred3_d,label = "Regular poly",s=15)
plt.title("Residual of the cheby and poly fit")
plt.legend()
plt.show()