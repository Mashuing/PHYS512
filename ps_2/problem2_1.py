import numpy as np

def lorentz(x):
    return 1/(1+x**2)

    
def integrate_step(fun,x1,x2,tol,s,x_f,y_f,c):
# add i to count the loop, and count to count the saving calculation
    i=s
    count=c
# print the integrate range, i, xi and yi in the last loop
    print('integrating from ',x1,' to ',x2,"i=",i,"xi=",x_f,"yi=",y_f)
# if is the first loop, just calculate the function value
    if i ==0:       
        xi=np.linspace(x1,x2,5)
        yi=fun(xi)
# if is not the first loop, judge if the value calculated before, if is, just use the value instead of call the function again.
    else:
        xi=x_f
        yi=y_f
        xii=np.linspace(x1,x2,5)
        yii=[]
        for a in range(len(xii)):
# if a value is not called function, count plus one
            if xii[a]==xi[a]:
                yii.append(yi[a])
                count =count+1
            else:
                yii.append(fun(xii[a]))
# Replace xi, yi with the current value
        xi=xii
        yi=yii
    area1=(x2-x1)*(yi[0]+4*yi[2]+yi[4])/6
    area2=(x2-x1)*( yi[0]+4*yi[1]+2*yi[2]+4*yi[3]+yi[4])/12
    myerr=np.abs(area1-area2)
    i=i+1
    print("error=",myerr)
# record the area, and the count which is the number of saved function calls
    if myerr<tol:
        return area2, count
    else:
# save the xi, yi and the count in the loop
        xm=0.5*(x1+x2)
        a1=integrate_step(fun,x1,xm,tol/2,i,xi,yi,count)[0]
        a2=integrate_step(fun,xm,x2,tol/2,i,xi,yi,count)[0]
        c1=integrate_step(fun,x1,xm,tol/2,i,xi,yi,count)[1]
        c2=integrate_step(fun,xm,x2,tol/2,i,xi,yi,count)[1]
        return a1+a2, c1+c2

ans=integrate_step(lorentz,-10,10,0.001,0,[],[],0)
print("Integrate Result=",ans[0],"saving function calls= ", ans[1])