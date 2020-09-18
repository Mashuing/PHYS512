# Problem 1
import numpy as np

expvals=np.linspace(-12,0,25)
x0=1
truth1=np.exp(x0)
truth2=0.01*np.exp(0.01*x0)
f0_1=np.exp(x0)
f0_2=np.exp(0.01*x0)
print("exp(x)")
for myexp in expvals:
    #print(myexp)
    dx=10**myexp
    f1=np.exp(x0+dx)
    f2=np.exp(x0-dx) #go from df/dx=f(x+dx)-f(x) to (f(x+dx)-f(x-dx))/2dx
    f3=np.exp(x0+2*dx)
    f4=np.exp(x0-2*dx)
    deriv=(f1-f0_1)/dx  #make the derivative from (f(x+dx)-f(x))/dx
    deriv2=(f1-f2)/(2*dx) #make the derivative out of (f(x+dx)-f(x-dx))/2dx
    deriv3 =(8*(f1-f2)-(f3-f4))/(12*dx)# make the derication out of (8(f1-f2)-(f3-f4))/12dx

    print("epilson value:",myexp,"Difference to the truth:",np.abs(deriv3-truth1))    
print("exp(0.01x)")
for myexp in expvals:
    #print(myexp)
    dx=10**myexp
    f1=np.exp(0.01*(x0+dx))
    f2=np.exp(0.01*(x0-dx)) #go from df/dx=f(x+dx)-f(x) to (f(x+dx)-f(x-dx))/2dx
    f3=np.exp(0.01*(x0+2*dx))
    f4=np.exp(0.01*(x0-2*dx))
    deriv=(f1-f0_2)/dx  #make the derivative from (f(x+dx)-f(x))/dx
    deriv2=(f1-f2)/(2*dx) #make the derivative out of (f(x+dx)-f(x-dx))/2dx
    deriv3 =(8*(f1-f2)-(f3-f4))/(12*dx)# make the derication out of (8(f1-f2)-(f3-f4))/12dx

    print("epilson value:",myexp,"Difference to the truth:",np.abs(deriv3-truth2))