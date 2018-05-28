import numpy as np
from matplotlib import pyplot as plt
from src.oreilly.my.common.mnist import *
from src.oreilly.my.common.myUtil import gradient_descent, numeric_gradient

def gradient(x):
    return 2 * x

def f(x):
#     print('x=', x, ', f=', np.sum(x**2))
    return np.sum(x**2)

if __name__ == '__main__':
    x = np.array([-3.0, 4.0])
    xlist = [x[0]]
    ylist = [x[1]]

    for i in np.arange(100):
#         x = gradient_descent(gradient, x, 0.1)
        x -= numeric_gradient(f, x, 0.1)
        xlist.append(x[0])
        ylist.append(x[1])

    plt.scatter(xlist, ylist)
    plt.show()