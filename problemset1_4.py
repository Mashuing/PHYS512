#Problem 4 code
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

def Efield(x,z):
    return (z-x)/(1-x**2+(z-x)**2)**1.5
# The field of a ring with charge when ignore the constant , and set R =1.
def integrate_step(z,x1,x2,tol):
    print('integrating from ',x1,' to ',x2)
    x=np.linspace(x1,x2,5)
    y=(z-x)/(1-x**2+(z-x)**2)**1.5
# The field of a ring with charge when ignore the constant , and set R =1.
    area1=(x2-x1)*(y[0]+4*y[2]+y[4])/6
    area2=(x2-x1)*( y[0]+4*y[1]+2*y[2]+4*y[3]+y[4])/12
    myerr=np.abs(area1-area2)
    if myerr<tol:
        return area2
    else:
        xm=0.5*(x1+x2)
        a1=integrate_step(z,x1,xm,tol/2)
        a2=integrate_step(z,xm,x2,tol/2)
        return a1+a2


#x^3
#ans=integrate_step(x3,0,1,0.0001)
#print(ans)
#assert(1==0)
d = np.linspace(0,10,1001)
x0=-1
x1=1
E = []
# Seperate the z=R, using integrator
for z in d:
    if z==1:
        E.append(0)
    else:
        ans=integrate_step(z,x0,x1,0.0001)
        E.append(ans)
    
#     if z==1:
#         E.append(0)
#     else:
#         # ans=integrate_step(z,x0,x1,0.0001)
#         fun = lambda x:(z-x)/(1-x**2+(z-x)**2)**1.5
#         ans=integrate.quad(fun,x0,x1)[0]
#         E.append(ans)
# Not seperate the z=R, using scipy.integrate
# for z in d:
#     # ans=integrate_step(z,x0,x1,0.0001)
#     fun = lambda x:(z-x)/(1-x**2+(z-x)**2)**1.5
#     ans=integrate.quad(fun,x0,x1)[0]
#     E.append(ans)
plt.plot(d,E)

