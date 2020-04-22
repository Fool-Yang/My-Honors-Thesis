from numpy import array
import matplotlib
import matplotlib.pyplot as plt
font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 24}
matplotlib.rc('font', **font)

from machine import Machine

# y = x^2
M = Machine((1), nodes = 4096, lr = 0.02)
X = [1, 2, 3, 4, 5, 6, 7, 20, 21, 22, 23, 24, 25, 26]
Y = [x**2 for x in X]
M.learn(X, Y, 2048)
M.save("x^2")
X_test = [1 + 25/1000*x for x in range(1001)]
Y_test = M.v(X_test)
Y_t = [x**2 for x in X_test]
# plot
fig, ax = plt.subplots()
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("y = x^2")
ax.plot(X_test, Y_test, label = "prediction")
ax.plot(X_test, Y_t, label = "actrual")
ax.plot(X, Y, 'o', label = "sample")
plt.legend(loc='best')

#M = Machine(model = 'x^2')
X = [[x] for x in range(1, 27)]
dx = M.d(X)[0]
Mdx = Machine((1), nodes = 512, lr = 0.04)
Mdx.learn(X, dx, 512)
Mdx.save("2x")

Y_test = Mdx.v(X_test)
Y_t = [2*x for x in X_test]
# plot
fig, ax = plt.subplots()
ax.set_xlabel("x")
ax.set_ylabel("y'")
ax.set_title("first derivative")
ax.plot(X_test, Y_test, label = "prediction")
ax.plot(X_test, Y_t, label = "actrual")
plt.legend(loc='best')

dxx = Mdx.d(X)[0]
Mdxx = Machine((1), nodes = 512, lr = 0.04)
Mdxx.learn(X, dxx, 512)
Mdxx.save("2")

Y_test = Mdxx.v(X_test)
Y_t = [2 for x in X_test]
# plot
fig, ax = plt.subplots()
ax.set_xlabel("x")
ax.set_ylabel("y''")
ax.set_title("second derivative")
ax.plot(X_test, Y_test, label = "prediction")
ax.plot(X_test, Y_t, label = "actrual")
plt.legend(loc='best')

dxxx = Mdxx.d(X)[0]
Mdxxx = Machine((1), nodes = 512, lr = 0.04)
Mdxxx.learn(X, dxxx, 512)
Mdxxx.save("0")

Y_test = Mdxxx.v(X_test)
Y_t = [0 for x in X_test]
# plot
fig, ax = plt.subplots()
ax.set_xlabel("x")
ax.set_ylabel("y'''")
ax.set_title("third derivative")
ax.plot(X_test, Y_test, label = "prediction")
ax.plot(X_test, Y_t, label = "actrual")
plt.legend(loc='best')
plt.show()

X0 = [[x] for x in range(3, 3 + 5)]
print(M.v(X0), M.d(X0)[0], Mdx.d(X0)[0], Mdxx.d(X0)[0], sep = '\n')
