from numpy import array
import matplotlib
import matplotlib.pyplot as plt
font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 24}
matplotlib.rc('font', **font)

from machine import Machine

# y = 0.5x^2 - 2x + 1
M = Machine((1), nodes = 2048, lr = 0.0036)
X = array([1, 2, 3, 4, 5, 6, 7, 8, 9, 19, 20, 21, 22, 23, 24, 25, 26])
Y = array([0.5*x**2 - 2*x + 1 for x in X])
M.learn(X, Y, 4096)
M.save("0.5x^2 - 2x + 1")
X_test = array([[1 + 25/1000*x] for x in range(1001)])
Y_test = M.v(X_test)
Y_t = [0.5*x**2 - 2*x + 1 for x in X_test]
# plot
fig, ax = plt.subplots()
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("y = 0.5x^2 - 2x + 1")
ax.plot(X_test.ravel(), Y_test, label = "prediction")
ax.plot(X_test.ravel(), Y_t, label = "actual")
ax.plot(X, Y, 'o', label = "sample")
plt.legend(loc='best')



X = array([[x] for x in range(3, 25)])
dx = M.d(X)[0]
Mdx = Machine((1), nodes = 512, lr = 0.006)
Mdx.learn(X, dx, 512)
Mdx.save("x - 2")

X_test = array([[3 + 21/1000*x] for x in range(1001)])
Y_test = Mdx.v(X_test).ravel()
Y_t = [x - 2 for x in X_test]
# plot
fig, ax = plt.subplots()
ax.set_xlabel("x")
ax.set_ylabel("y'")
ax.set_title("first derivative")
ax.plot(X_test.ravel(), Y_test, label = "prediction")
ax.plot(X_test.ravel(), Y_t, label = "actual")
plt.legend(loc='best')



X = array([[x] for x in range(5, 23)])
dxx = Mdx.d(X)[0]
Mdxx = Machine((1), nodes = 512, lr = 0.01)
Mdxx.learn(X, dxx, 512)
Mdxx.save("1")

X_test = array([[5 + 17/1000*x] for x in range(1001)])
Y_test = Mdxx.v(X_test).ravel()
Y_t = [1 for x in X_test]
# plot
fig, ax = plt.subplots()
ax.set_xlabel("x")
ax.set_ylabel("y''")
ax.set_title("second derivative")
ax.plot(X_test.ravel(), Y_test, label = "prediction")
ax.plot(X_test.ravel(), Y_t, label = "actual")
plt.legend(loc='best')



X = array([[x] for x in range(7, 21)])
dxxx = Mdxx.d(X)[0]
Mdxxx = Machine((1), nodes = 512, lr = 0.01)
Mdxxx.learn(X, dxxx, 512)
Mdxxx.save("0")

X_test = array([[7 + 13/1000*x] for x in range(1001)])
Y_test = Mdxxx.v(X_test).ravel()
Y_t = [0 for x in X_test]
# plot
fig, ax = plt.subplots()
ax.set_xlabel("x")
ax.set_ylabel("y'''")
ax.set_title("third derivative")
ax.plot(X_test.ravel(), Y_test, label = "prediction")
ax.plot(X_test.ravel(), Y_t, label = "actual")
plt.legend(loc='best')
plt.show()



X0 = [[x] for x in range(7, 7 + 5)]
print(M.v(X0), Mdx.v(X0), Mdxx.v(X0), Mdxxx.v(X0), sep = '\n')
