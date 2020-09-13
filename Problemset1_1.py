# Problem 1
import numpy as np

expvals=np.linspace(-12,-1,23)
x0=1
#d(exp(x))/dx = exp(x)
#true derivative is just exp(x0)
truth=np.exp(x0)
f0=np.exp(x0)
for myexp in expvals:
    #print(myexp)
    dx=10**myexp
    f1=np.exp(x0+dx)
    f2=np.exp(x0-dx) #go from df/dx=f(x+dx)-f(x) to (f(x+dx)-f(x-dx))/2dx
    f3=np.exp(x0+2*dx)
    f4=np.exp(x0-2*dx)
    deriv=(f1-f0)/dx  #make the derivative from (f(x+dx)-f(x))/dx
    deriv2=(f1-f2)/(2*dx) #make the derivative out of (f(x+dx)-f(x-dx))/2dx
    deriv3 =(8*(f1-f2)-(f3-f4))/(12*dx)
    print(myexp,deriv,np.abs(deriv-truth),np.abs(deriv2-truth),np.abs(deriv3-truth))