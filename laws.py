from numpy.random import normal
from numpy import array
mu, sigma = 0, 0.001

# newton's law
# f = ma
def newton():
    m = list(map(float, range(5, 5 + 6)))
    a = list(map(float, range(3, 3 + 6)))
    m_a = [[x, y] for x in m for y in a]
    f = [ma[0]*ma[1] for ma in m_a]
    for i in range(len(m_a)):
        m_a[i][0] += m_a[i][0]*normal(mu, sigma)
        m_a[i][1] += m_a[i][1]*normal(mu, sigma)
    for i in range(len(f)):
        f[i] += f[i]*normal(mu, sigma)
    return array(m_a), array(f)
