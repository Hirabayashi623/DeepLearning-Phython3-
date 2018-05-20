import numpy as np
from src.oreilly.my.common import myUtil

def sigmoid(x):
    return 1 / ( 1 + np.exp(-x) )

def step(x):
    if( x <= 0):
        return 0
    else:
        return 1

def ReLU(x):
    if x <= 0:
        return 0
    else:
        return x

def softmax(x):
    # a / b
    max = np.max(x)
    a = np.array([np.exp(xi) - max for xi in x])
    b = np.sum(a)
    return a / b;

if(__name__ == '__main__'):
#     myUtil.autoplot(ReLU, -1, 1, 10000)
    a = np.array([1,2,3,4,5])
    print(softmax(a))
    print(np.sum(softmax(a)))