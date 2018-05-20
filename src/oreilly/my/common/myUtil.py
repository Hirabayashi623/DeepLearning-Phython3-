from matplotlib import pyplot as plt
import numpy as np

def autoplot(func, xrange1=-5, xrange2=5, sample=10000):
    xlist = np.arange(xrange1, xrange2, (xrange2-xrange1)/sample)
    ylist = np.array([func(x) for x in xlist])
    plt.plot(xlist, ylist)
    plt.show()
