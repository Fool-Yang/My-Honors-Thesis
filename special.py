import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib
font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 16}
matplotlib.rc('font', **font)
matplotlib.rcParams['legend.fontsize'] = 24

from laws import special
from machine import Machine

c = 15.0

m = np.arange(5.0, 10.0, 0.05)
a = np.arange(3.0, 8.0, 0.05)
X_test, Y_test = np.meshgrid(m, a)

# v = (u + w)/(1 + uw/c^2)
def F(m, a):
    return (m + a)/(1 + m*a/c**2)

M = Machine(model = "v")
'''
M = Machine((2), nodes = 1024, lr = 0.0006)
m_a, f = special()
xs = np.array([ma[0] for ma in m_a])
ys = np.array([ma[1] for ma in m_a])
zs_s = f
M.learn(m_a, f, 10000)
M.save("v")

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
ax.set_title("v = (u + w)/(1 + uw/c^2)")
ax.set_xlabel('u')
ax.set_ylabel('w')
ax.set_zlabel('v')
ax.legend()
'''
def F_m(m, a):
    return c**2*(c**2 - a**2)/(c**2 + m*a)**2

def F_a(m, a):
    return c**2*(c**2 - m**2)/(c**2 + m*a)**2

m = np.arange(6.0, 9.0, 0.05)
a = np.arange(4.0, 7.0, 0.05)
X_test, Y_test = np.meshgrid(m, a)
m_a = np.array(list(zip(np.ravel(X_test), np.ravel(Y_test))))

xs = np.arange(6.0, 9.0, 0.15)
ys = np.arange(4.0, 7.0, 0.15)
X = np.array([(x, y) for x in xs for y in ys])

dm, da = M.d(X)

# Mdm = Machine(model = "du")
Mdm = Machine((2), nodes = 1, lr = 0.00006)
Mdm.learn(X, dm, 8192)
Mdm.save("du")

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
ax.scatter(xs, ys, dm, label = "sample", color = "green")
ax.set_title("dv/du")
ax.set_xlabel('u')
ax.set_ylabel('w')
ax.set_zlabel('v_u')
ax.legend()
'''
Mda = Machine((2), nodes = 128, lr = 0.006)
Mda.learn(X, da, 4096)
Mda.save("dw")

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
ax.set_title("dv/dw")
ax.set_xlabel('u')
ax.set_ylabel('w')
ax.set_zlabel('v_w')
ax.legend()
'''
plt.show()
