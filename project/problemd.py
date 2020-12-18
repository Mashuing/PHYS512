import numpy as np
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import time
import pandas
import math
plt.rcParams['axes.prop_cycle'    ] = plt.cycler('color',['lightseagreen', 'indigo' ])
def greens(n,soft=0.03,G=0.001):
    #get the potential from a particule at (0,0,0)
    #Take G to 1
    x = np.arange(n)/n
    x[n//2:] = x[n//2:] - n/n
    # x= np.arange(-n//2,n//2)/n
    xx,yy,zz = np.meshgrid(x, x, x)
    pot = np.zeros([n,n,n])
    dr=np.sqrt(xx**2+yy**2+zz**2)
    # Set the softr
    dr[dr<soft] = soft
    pot=1*G/dr
    # pot=pot-pot[n//2,n//2,n//2]  #set it so the potential at the edge goes to zero

    return pot

# # Check if the potential for one particle in 3d is correct
# def wrap_for_image(img):
#     return np.roll(img, img.shape[0]//2, axis=(0,1))
# m =128
# pot = greens(2*m,5)
# pot2d = np.zeros([2*m,2*m])
# x = 0
# for i in range(2*m):
#     for j in range(2*m):
#         pot2d[i,j]=pot[i,j,0]       
# plt.imshow(wrap_for_image(pot2d))
# plt.colorbar()


def density(x,n,m):   
    bc,edges = np.histogramdd(x,bins=(n,n,n),range=((0,n),(0,n),(0,n)),weights=m)
    mask = np.zeros([n,n,n],dtype='bool')
    return bc, edges
def get_forces(pot,dx):
    grad = np.gradient(pot,dx)
    return grad
def get_forces_2(pot,deltax):
    dx,dy,dz = np.gradient(pot,deltax)
    return dx,dy,dz
def rho2pot(rho,kernelft):
    tmp=rho.copy()
    # tmp=np.pad(tmp,(0,tmp.shape[0]))

    tmpft=np.fft.rfftn(tmp)
    tmp=np.fft.irfftn(tmpft*kernelft)
    # if len(rho.shape)==2:
    #     tmp=tmp[:rho.shape[0],:rho.shape[1]]
    #     return tmp
    if len(rho.shape)==3:
        tmp=tmp[:rho.shape[0],:rho.shape[1],:rho.shape[2]]
        return tmp
    print("error in rho2pot - unexpected number of dimensions")
    assert(1==0)

# def rho2pot_masked(rho,mask,kernelft,return_mat=False):
#     rhomat=np.zeros(mask.shape)
#     rhomat[mask]=rho
#     potmat=rho2pot(rhomat,kernelft)
#     if return_mat:
#         return potmat
#     else:
#         return potmat[mask]
def take_step(x,v,dt,n,kernelft,m):
    xx=x+0.5*v*dt
    # Periodical condition
    xx[xx<=0.5] = n-0.5
    xx[xx>=n-0.5]= 0.5
    den = density(xx,n,m)[0]
    pot = rho2pot(den,kernelft)
    # ff=np.asarray(get_forces(pot))
    f = np.zeros(xx.shape)
    ff = get_forces_2(pot,1/n)
    ffx = ff[0]
    ffy = ff[1]
    ffz = ff[2]
    for i in range(xx.shape[0]):
        xx_int = np.rint(xx[i])
        # fx = ff[0][int(xx_int[0]),int(xx_int[1]),int(xx_int[2])]/m[i]
        # fy = ff[1][int(xx_int[0]),int(xx_int[1]),int(xx_int[2])]/m[i]
        # fz = ff[2][int(xx_int[0]),int(xx_int[1]),int(xx_int[2])]/m[i]
        fx = ffx[int(xx_int[0]),int(xx_int[1]),int(xx_int[2])]
        fy = ffy[int(xx_int[0]),int(xx_int[1]),int(xx_int[2])]
        fz = ffz[int(xx_int[0]),int(xx_int[1]),int(xx_int[2])]
        f[i] = [fx,fy,fz]
    vv=v+0.5*dt*f
    x=x+dt*vv
    v=v+dt*f
    # print(f)
    # print("den=",den)
    # print("pot=",pot)
    # print("Force",f)
    return  f, x, v, den, pot, ff
def energy(x,v,pot,m):
    potential = np.sum(pot)
    vx= v[:,0]
    vy= v[:,1]
    vz = v[:,2]
    v_abs = vx**2+vy**2+vz**2
    kin = np.sum(0.5*m*v_abs)
    return potential+kin
# Convert x in grid to the real position
def covpos(x,n):
    x_r = x/n
    return x_r
# transfer the speed in grid to the real speed
def covvol(v,n):
    v_r = v/np.sqrt(n)
    return v_r
def k3(n):
    x = np.arange(n)
    # x= np.arange(-n//2,n//2)/n
    xx,yy,zz = np.meshgrid(x, x, x)
    scale = np.zeros([n,n,n])
    dk=np.sqrt(xx**2+yy**2+zz**2)
    soft = 0.6
    dk[dk<soft]=soft
    # Set the softr
    scale=1/dk**3
    # pot=pot-pot[n//2,n//2,n//2]  #set it so the potential at the edge goes to zero

    return scale
def getmass_new(mass,den,bins,x,n):
    denft = np.fft.fftn(den)
    ind = np.ones(x.shape)
    indx = np.digitize(x[:,0],bins[0])-1
    indy = np.digitize(x[:,1],bins[1])-1
    indz = np.digitize(x[:,2],bins[2])-1
    ind[:,0] = indx
    ind[:,1] = indy
    ind[:,2] = indz
    k_noise = k3(n)
    noise_den =np.fft.fftshift((np.fft.irfftn((k_noise))))
    mass_new = noise_den[indx,indy,indz]
    return k_noise, denft, noise_den,mass_new
    
# grid size
n =256
# number of particles
N =100000
dt=0.05
oversample = 3
T=0

x = n*np.random.rand(N,3)
v = 0*np.random.rand(N,3)

# x= np.array([[64,64,64],[64,44,64]])
# v=np.array([[0.,0.,0],[40.47,0,0]])
# x= np.array([[16,16,16]])
# v=np.array([[0.,0.,0]])

print(x)
# Set the mass
m=1*np.ones(N)
m[1]=1

kernel=greens(n)
kernelft=np.fft.rfftn(kernel)
den,bins = density(x,n,m)
k_noise,denft, noise_den, mass_new = getmass_new(m,den,bins,x,n)
mass_min = np.min(mass_new)
if mass_min<= 0:
    print("Error!! Negative mass")
    a=b
m=mass_new/mass_min
pot = rho2pot(den,kernelft)
# # # Check the force plot
# # den = density(x,n,m)[0]
# # pot = rho2pot(den,kernelft)
# nf = n
# # dx,dy,dz = get_forces_2(pot,1/n)
# # x = np.arange(nf)
# # X, Y = np.meshgrid(x, x)
# # dx2d=np.zeros([nf,nf])
# # dy2d=np.zeros([nf,nf])
# # dz2d=np.zeros([nf,nf])
# # for i in range(nf):
# #     for j in range(nf):
# #           dx2d[i,j]=dx[i,j,nf//2]
# #           dy2d[i,j]=dy[i,j,nf//2]
# #           dz2d[i,j]=dz[i,j,nf//2]

# # fig1, ax1 = plt.subplots()
# # fig2, ax2 = plt.subplots()
# # zero=np.zeros(dy.shape)
# # ax2.imshow(dx[:,:,nf//2])
# # ax1.quiver(Y, X,dx[:,:,nf//2],dy[:,:,nf//2])

# # ax1.xaxis.set_ticks([])
# # ax1.yaxis.set_ticks([])
# # ax1.set_aspect('equal')
# # plt.show()
# # # r = take_step(x,v,dt,n,kernelft,m)
# # Check if the potential for one particle in 3d is correct
# fig1, ax1 = plt.subplots()
# fig2, ax2 = plt.subplots()
# fig3, ax3 = plt.subplots()
# den2d = np.zeros([nf,nf])
# noiseden2d = np.zeros([nf,nf])
# x = 0
# for i in range(nf):
#     for j in range(nf):
#         noiseden2d[i,j]=noise_den[i,j,nf//2]     
#         den2d[i,j]=den[i,j,nf//2]       
# den2=ax2.imshow(noiseden2d)
# den1=ax1.imshow(den2d)
# fig1.colorbar(den1, ax=ax1)
# fig2.colorbar(den2, ax=ax2)
# ax3.set_title('particle masses')
# ax3.hist(mass_new, 50, log=True)
# ax3.set_xlabel('Mass')
# ax3.set_ylabel('Frequency')
# plt.show()

# Start the simulation
fig4=plt.figure()#Create 3D axes
ax=fig4.add_subplot(projection="3d")
# ax.scatter(x[:,0],x[:,1],x[:,2],color='blue',marker=".",s=0.02)
# ax.set_xlim(0,n)
# ax.set_ylim(0,n)
# ax.set_zlim(0,n)
# for i in range (x.shape[0]):
    
#     ax.scatter(x[i][0],x[i][1],x[i][2],color='blue',marker=".",s=0.02)
#     ax.set_xlim(0,n)
#     ax.set_ylim(0,n)
#     ax.set_zlim(0,n)
# fig.savefig('D:/git_code/PHYS512/project/ps_c/3dinitial.png', dpi=600)
a=0
position = []
f= open('D:\git_code\PHYS512\project\ps_d\position.npy','ab')
ener = []
time = []

for t in range(20000):
     T=T+dt
     position.append(x)
     np.save(f,position)
     E = energy(x,v,pot,m)
     ener.append(E)
     time.append(T)
     if t%oversample==0:
         plt.cla()
         ax.scatter(x[:,0],x[:,1],x[:,2],color='blue',marker=".",s=0.02)
         ax.set_xlim(0,n)
         ax.set_ylim(0,n)
         ax.set_zlim(0,n)
         ax.set_title('Time ='+str(T)+"\nEnergy="+str(E))
         fig4.savefig('D:/git_code/PHYS512/project/ps_d/'+'bla'+str(t)+'.png', dpi=600)
         
                
         plt.pause(0.001)
     np.savetxt('D:\git_code\PHYS512\project\ps_d\Energy.txt',ener)
     x_tmp= x
     v_tmp = v
     step = take_step(x_tmp,v_tmp,dt,n,kernelft,m)
     x_new = step[1]
     v_new = step[2]
     pot = step[4]
    
     x= x_new
     v = v_new
     x[x<0.5] = n-0.5
     x[x>n-0.5]=0.5
     print(t)

     a=a+1
     print(E)



