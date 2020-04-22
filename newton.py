import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['legend.fontsize'] = 20

from laws import newton
from machine import Machine

# F = ma
def F(m, a):
    return m*a

M = Machine((2), nodes = 2048, lr = 0.002)
m_a, f = newton()
xs = np.array([ma[0] for ma in m_a])
ys = np.array([ma[1] for ma in m_a])
zs_s = f
M.learn(m_a, f, 4096)
M.save("ma")

m = np.arange(5.0, 10.0, 0.05)
a = np.arange(3.0, 8.0, 0.05)
X_test, Y_test = np.meshgrid(m, a)
m_a = np.array(list(zip(np.ravel(X_test), np.ravel(Y_test))))

zs_t = np.array(F(np.ravel(X_test), np.ravel(Y_test)))
Z_t = zs_t.reshape(X_test.shape)

zs = M.v(m_a)
Z = zs.reshape(X_test.shape)

# plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X_test, Y_test, Z, label = "prediction")
surf._facecolors2d=surf._facecolors3d
surf._edgecolors2d=surf._edgecolors3d
surf = ax.plot_surface(X_test, Y_test, Z_t, label = "actrual")
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

xs = np.arange(5.0, 10.0, 0.2)
ys = np.arange(3.0, 8.0, 0.2)
X = np.array([(x, y) for x in xs for y in ys])

dm, da = M.d(X)

Mdm = Machine((2), nodes = 1024, lr = 0.00001)
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
surf = ax.plot_surface(X_test, Y_test, Z_t, label = "actrual")
surf._facecolors2d=surf._facecolors3d
surf._edgecolors2d=surf._edgecolors3d
ax.set_title("dF/dm")
ax.set_xlabel('m')
ax.set_ylabel('a')
ax.set_zlabel('F_m')
ax.legend()

Mda = Machine((2), nodes = 1024, lr = 0.00001)
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
surf = ax.plot_surface(X_test, Y_test, Z_t, label = "actrual")
surf._facecolors2d=surf._facecolors3d
surf._edgecolors2d=surf._edgecolors3d
ax.set_title("dF/da")
ax.set_xlabel('m')
ax.set_ylabel('a')
ax.set_zlabel('F_a')
ax.legend()

plt.show()
'''
X0 = [[x] for x in range(3, 3 + 5)]
print(M.v(X0), M.d(X0)[0], Mdx.d(X0)[0], Mdxx.d(X0)[0], sep = '\n')
'''
