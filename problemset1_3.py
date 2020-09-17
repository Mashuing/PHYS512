import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
def cos(x):
    return np.cos(x)
def lorentz(x):
    return 1/(1+x**2)
def func(f,x):
    a=f
    if a==0:
        y =np.cos(x)
        return y
    else:
        z = 1/(1+x**2)
        return z
num=1
#For cos(x)
# xi = np.linspace(-0.5*np.pi,0.5*np.pi,10)

# fun = func(num,xi)
#For Lorentzian
xi = np.linspace(-1,1,10)
fun = func(num,xi)
# polynomial 1st
print("polynomial 1st")
x = np.linspace(xi[1],xi[-1],1001)
y_true = np.cos(x)
y_interp = np.zeros(len(x))
for i in range(len(x)):        
    ind=np.max(np.where(x[i]>=xi)[0])
    x_use=xi[ind-1:ind+1]
    y_use=fun[ind-1:ind+1]
    pars=np.polyfit(x_use,y_use,1)
    pred=np.polyval(pars,x[i])
    y_interp[i]=pred
fig1 =plt.figure()
plt.plot(xi,fun,"*",label = "True value points")
plt.plot(x,y_interp,label = "First order polynomial")
print("error =",np.std(y_true-y_interp))
# polynomial 2nd
print("polynomial 2nd")
x = np.linspace(xi[1],xi[-2],1001)
y_true = func(num,x)
y_interp = np.zeros(len(x))
for i in range(len(x)):        
    ind=np.max(np.where(x[i]>=xi)[0])
    x_use=xi[ind-1:ind+2]
    y_use=fun[ind-1:ind+2]
    pars=np.polyfit(x_use,y_use,2)
    pred=np.polyval(pars,x[i])
    y_interp[i]=pred
plt.plot(x,y_interp,label = "Second order polynomial")
print("error =",np.std(y_true-y_interp))
# polynomial 3rd
print("polynomial 3rd")
x = np.linspace(xi[1],xi[-3],1001)
y_true = func(num,x)
y_interp = np.zeros(len(x))
for i in range(len(x)):        
    ind=np.max(np.where(x[i]>=xi)[0])
    x_use=xi[ind-1:ind+3]
    y_use=fun[ind-1:ind+3]
    pars=np.polyfit(x_use,y_use,3)
    pred=np.polyval(pars,x[i])
    y_interp[i]=pred
print("error =",np.std(y_true-y_interp))
plt.plot(x,y_interp,label = "Third order polynomial")
# cubic spline
print("Cubic Spline")
x = np.linspace(xi[0],xi[-1],1001)
y_true = func(num,x)
y_interp = np.zeros(len(x))
f = interpolate.interp1d(xi,fun,kind = "cubic")
y_interp = f(x)
plt.plot(x,y_interp,label = "Cubic Spline")
print("error =",np.std(y_true-y_interp))
# rational function
print("Rational Function")
def rat_eval(p,q,x):
    top=0
    for i in range(len(p)):
        top=top+p[i]*x**i
    bot=1
    for i in range(len(q)):
        bot=bot+q[i]*x**(i+1)
    return top/bot

def rat_fit(x,y,n,m):
    assert(len(x)==n+m-1)
    assert(len(y)==len(x))
    mat=np.zeros([n+m-1,n+m-1])
    for i in range(n):
        mat[:,i]=x**i
    for i in range(1,m):
        mat[:,i-1+n]=-y*x**i
    pars=np.dot(np.linalg.inv(mat),y)
    #pars=np.dot(np.linalg.pinv(mat),y)
    p=pars[:n]
    q=pars[n:]
    return p,q


#1*p0 + x*p1 +x**2+p2+... -q1*x - q2*x**2... = y

n=5
m=6

p,q=rat_fit(xi,fun,n,m)
x = np.linspace(-5*xi[0],5*xi[-1],1001)
y_true = func(num,x)
y_interp = np.zeros(len(x))
y_interp=rat_eval(p,q,x)
plt.plot(x,y_interp,label="Rational Function")

print("error =",np.std(y_true-y_interp),p,q)
plt.legend()
plt.show()


