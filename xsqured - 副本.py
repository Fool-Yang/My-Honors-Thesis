from numpy import array
import matplotlib
import matplotlib.pyplot as plt
font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 24}
matplotlib.rc('font', **font)

from machine import Machine

# y = x^2
M = Machine(model = "x^2")
X_test = [1 + 25/1000*x for x in range(1001)]
Y_test = M.v(X_test)
Y_t = [x**2 for x in X_test]
print(sum([abs(y_test - y_t) for (y_test, y_t) in zip(Y_test, Y_t)])/len(X_test))
