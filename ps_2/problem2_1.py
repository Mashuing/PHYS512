import numpy as np

def lorentz(x):
    return 1/(1+x**2)

    
def integrate_step(fun,x1,x2,tol,s,x_f,y_f,c):
    i=s
    count=c
    print('integrating from ',x1,' to ',x2,"i=",i,"xi=",x_f,"yi=",y_f)
    if i ==0:       
        xi=np.linspace(x1,x2,5)
        yi=fun(xi)
    else:
        xi=x_f
        yi=y_f
        xii=np.linspace(x1,x2,5)
        yii=[]
        for a in range(len(xii)):
            if xii[a]==xi[a]:
                yii.append(yi[a])
                count =count+1
            else:
                yii.append(fun(xii[a]))
        xi=xii
        yi=yii
    area1=(x2-x1)*(yi[0]+4*yi[2]+yi[4])/6
    area2=(x2-x1)*( yi[0]+4*yi[1]+2*yi[2]+4*yi[3]+yi[4])/12
    myerr=np.abs(area1-area2)
    i=i+1
    print("error=",myerr)
    if myerr<tol:
        return area2, count
    else:
        xm=0.5*(x1+x2)
        a1=integrate_step(fun,x1,xm,tol/2,i,xi,yi,count)[0]
        a2=integrate_step(fun,xm,x2,tol/2,i,xi,yi,count)[0]
        c1=integrate_step(fun,x1,xm,tol/2,i,xi,yi,count)[1]
        c2=integrate_step(fun,xm,x2,tol/2,i,xi,yi,count)[1]
        return a1+a2, c1+c2

ans=integrate_step(lorentz,-10,10,0.001,0,[],[],0)
print("Integrate Result=",ans[0],"saving function calls= ", ans[1])