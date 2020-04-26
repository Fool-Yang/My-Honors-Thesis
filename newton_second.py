import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib
font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 16}
matplotlib.rc('font', **font)
matplotlib.rcParams['legend.fontsize'] = 24

from laws import newton
from machine import Machine

m = np.arange(7.0, 8.0, 0.02)
a = np.arange(7.0, 8.0, 0.02)
X_test, Y_test = np.meshgrid(m, a)
m_a = np.array(list(zip(np.ravel(X_test), np.ravel(Y_test))))

xs = np.arange(7.0, 8.0, 0.1)
ys = np.arange(7.0, 8.0, 0.1)
X = np.array([(x, y) for x in xs for y in ys])

Mdm = Machine(model='dm')

dmm, dma = Mdm.d(X)

Mdmm = Machine((2), nodes = 8, lr = 0.1)
Mdmm.learn(X, dmm, 2048)
Mdmm.save("dmm")

zs = Mdmm.v(m_a)
Z = zs.reshape(X_test.shape)

zs_t = np.array([0.0] * (len(X_test) * len(X_test[0])))
Z_t = zs_t.reshape(X_test.shape)

# plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X_test, Y_test, Z, label = "prediction")
surf._facecolors2d=surf._facecolors3d
surf._edgecolors2d=surf._edgecolors3d
surf = ax.plot_surface(X_test, Y_test, Z_t, label = "actual")
surf._facecolors2d=surf._facecolors3d
surf._edgecolors2d=surf._edgecolors3d
ax.set_title("ddF/dmdm")
ax.set_xlabel('m')
ax.set_ylabel('a')
ax.set_zlabel('F_mm')
ax.legend()

plt.show()

M = Machine(model = "ma")
Mdm = Machine(model = "dm")
Mda = Machine(model = "da")
Mdmm = Machine(model = "dmm")
Mdma = Machine(model = "dma")
Mdam = Machine(model = "dam")
Mdaa = Machine(model = "daa")

X0 = np.array([[7.0, 5.0], [7.0, 6.0], [8.0, 5.0], [8.0, 6.0], [7.5, 5.5]])

print(M.v(X0), Mdm.v(X0), Mda.v(X0), Mdmm.v(X0), Mdma.v(X0), Mdam.v(X0), Mdaa.v(X0), sep = '\n')
