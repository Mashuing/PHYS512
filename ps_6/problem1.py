import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


# load some test data for demonstration and plot a wireframe
datas =np.loadtxt('rand_points.txt')
X, Y, Z = datas[:,0], datas[:,1], datas[:,2]
ax.scatter3D(X,Y,Z)

# # rotate the axes and update
# for angle in range(0, 360,5):
#     ax.view_init(90, angle)
#     plt.draw()
#     plt.pause(.001)
#     print(angle)