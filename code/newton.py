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

m = np.arange(5.0, 10.0, 0.05)
a = np.arange(3.0, 8.0, 0.05)
X_test, Y_test = np.meshgrid(m, a)

# F = ma
def F(m, a):
    return m*a

# M = Machine(model = "ma")
M = Machine((2), nodes = 4096, lr = 0.0024)
m_a, f = newton()
xs = np.array([ma[0] for ma in m_a])
ys = np.array([ma[1] for ma in m_a])
zs_s = f
M.learn(m_a, f, 4096)
M.save("ma")

zs_t = np.array(F(np.ravel(X_test), np.ravel(Y_test)))
Z_t = zs_t.reshape(X_test.shape)

m_a = np.array(list(zip(np.ravel(X_test), np.ravel(Y_test))))
zs = M.v(m_a)
Z = zs.reshape(X_test.shape)

# plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X_test, Y_test, Z, label = "prediction")
surf._facecolors2d=surf._facecolors3d
surf._edgecolors2d=surf._edgecolors3d
surf = ax.plot_surface(X_test, Y_test, Z_t, label = "actual")
surf._facecolors2d=surf._facecolors3d
surf._edgecolors2d=surf._edgecolors3d
ax.scatter(xs, ys, zs_s, label = "sample", color = "green")
ax.set_title("F = ma")
ax.set_xlabel('m')
ax.set_ylabel('a')
ax.set_zlabel('F')
ax.legend()

def F_m(m, a):
    return a

def F_a(m, a):
    return m

m = np.arange(6.0, 9.0, 0.05)
a = np.arange(4.0, 7.0, 0.05)
X_test, Y_test = np.meshgrid(m, a)
m_a = np.array(list(zip(np.ravel(X_test), np.ravel(Y_test))))

xs = np.arange(6.0, 9.0, 0.2)
ys = np.arange(4.0, 7.0, 0.2)
X = np.array([(x, y) for x in xs for y in ys])

dm, da = M.d(X)

# Mdm = Machine(model = "dm")
Mdm = Machine((2), nodes = 8, lr = 0.006)
Mdm.learn(X, dm, 4096)
Mdm.save("dm")

zs_t = np.array(F_m(np.ravel(X_test), np.ravel(Y_test)))
Z_t = zs_t.reshape(X_test.shape)

zs = Mdm.v(m_a)
Z = zs.reshape(X_test.shape)

# plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X_test, Y_test, Z, label = "prediction")
surf._facecolors2d=surf._facecolors3d
surf._edgecolors2d=surf._edgecolors3d
surf = ax.plot_surface(X_test, Y_test, Z_t, label = "actual")
surf._facecolors2d=surf._facecolors3d
surf._edgecolors2d=surf._edgecolors3d
ax.set_title("dF/dm")
ax.set_xlabel('m')
ax.set_ylabel('a')
ax.set_zlabel('F_m')
ax.legend()

Mda = Machine((2), nodes = 8, lr = 0.006)
Mda.learn(X, da, 4096)
Mda.save("da")

zs_t = np.array(F_a(np.ravel(X_test), np.ravel(Y_test)))
Z_t = zs_t.reshape(X_test.shape)

zs = Mda.v(m_a)
Z = zs.reshape(X_test.shape)

# plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X_test, Y_test, Z, label = "prediction")
surf._facecolors2d=surf._facecolors3d
surf._edgecolors2d=surf._edgecolors3d
surf = ax.plot_surface(X_test, Y_test, Z_t, label = "actual")
surf._facecolors2d=surf._facecolors3d
surf._edgecolors2d=surf._edgecolors3d
ax.set_title("dF/da")
ax.set_xlabel('m')
ax.set_ylabel('a')
ax.set_zlabel('F_a')
ax.legend()

plt.show()
