import numpy as np

from machine import Machine

M = Machine(model = "0.5x^2 - 2x + 1")
X_test = np.array([[1 + 25/1000*x] for x in range(1001)])
Y_test = M.v(X_test)
Y_t = np.array([0.5*x**2 - 2*x + 1 for x in X_test])
Loss = sum(abs(Y_t - Y_test))
print(Loss/len(Y_test))

def F(m, a):
    return m*a
m = np.arange(5.0, 10.0, 0.05)
a = np.arange(3.0, 8.0, 0.05)
X_test, Y_test = np.meshgrid(m, a)
M = Machine(model = "ma")
zs_t = np.array(F(np.ravel(X_test), np.ravel(Y_test)))
Z_t = zs_t.ravel()
m_a = np.array(list(zip(np.ravel(X_test), np.ravel(Y_test))))
zs = M.v(m_a)
Z = zs.ravel()
Loss = sum(abs(Z_t - Z))
print(Loss/len(Z_t))

'''
0.354
0.135
'''
