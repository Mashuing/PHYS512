import numpy as np
import matplotlib.pyplot as plt
datas = np.loadtxt("lakeshore.txt") # load the data
voltage = datas[:,1]  # load the voltage data
temp =  datas[:,0]    # load the temperature data
  # take the odd number data to do the interpolation
voltage_s = voltage[::2]
temp_s = temp[::2]
v=np.linspace(voltage_s[1],voltage_s[-1],1440) #take the point
t_interp = np.zeros(len(v))
t_checkerror = np.zeros(len(voltage))

for i in range(len(v)):
    ind=np.max(np.where(v[i]<=voltage_s)[0])# find the point to the left
    e = np.max(np.where(v[i]<=voltage)[0])#find the nearerst point to check error
    v_use = voltage_s[ind-1:ind+3] # choose the 3 point to fit
    t_use = temp_s[ind-1:ind+3]
    pars = np.polyfit(v_use,t_use,3)
    pred = np.polyval(pars,v[i])
    interp = np.polyval(pars,voltage[e])
    t_checkerror[e]=interp
    t_interp[i] = pred
t_checkerror[-1]=temp[-1]    #set the uninterpulte point to the same
plt.plot(voltage, temp,"*")
plt.plot(v,t_interp)
plt.plot(voltage,t_checkerror,"*")
print("error =",np.std(temp-t_checkerror))