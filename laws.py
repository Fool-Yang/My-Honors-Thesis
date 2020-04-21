from numpy.random import normal
from numpy import array
mu, sigma = 0, 0.001

# newton's law
# f = ma
def newton():
    m = array(list(map(float, range(5, 5 + 15))))
    a = array(list(map(float, range(1, 1 + 15))))
    f = m*a
    m += m*normal(mu, sigma)
    a += a*normal(mu, sigma)
    f += f*normal(mu, sigma)
    return m, a, f
