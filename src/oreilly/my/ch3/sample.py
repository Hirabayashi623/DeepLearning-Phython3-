import numpy as np
from src.oreilly.my.common import myUtil

def sigmoid(x):
    return 1 / ( 1 + np.exp(-x) )

# def step(x):
#     if( x <= 0):
#         return 0
#     else:
#         return 1

def step(x):
    ary = np.arange(x.size)
    ary[x>0] = 1
    ary[x<=0] = 0
    return ary

def ReLU(x):
    if x <= 0:
        return 0
    else:
        return x

def softmax(x):
    # a / b
    max = np.max(x)
    a = np.array([np.exp(xi-max) for xi in x])
    b = np.sum(a)
    return a / b;

if(__name__ == '__main__'):
     myUtil.autoplot(step, -1, 1, 10000)
#     a = np.array([1,2,3,4,5])
#     print(softmax(a))
#     print(np.sum(softmax(a)))