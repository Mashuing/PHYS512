import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import random

fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')


# load some test data for demonstration and plot a wireframe
datas =np.loadtxt('rand_points.txt')
X, Y, Z = datas[:,0], datas[:,1], datas[:,2]
ax1.scatter3D(X,Y,Z)

# # rotate the axes and update
# for angle in range(0, 360,5):
#     ax1.view_init(45, angle)
#     plt.draw()
#     plt.pause(.001)
#  Same proecess with python random number genetrator 
A = []
B = []
C = []
for i in range(20000):    
    x = random.randrange(0,10**8)
    y = random.randrange(0,10**8)
    z = random.randrange(0,10**8)
    A.append(x)
    B.append(y)
    C.append(z)
fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
ax2.scatter3D(A,B,C)
# for angle in range(0, 360,5):
#     ax2.view_init(45, angle)
#     plt.draw()
#     plt.pause(.001)
#     print(angle)