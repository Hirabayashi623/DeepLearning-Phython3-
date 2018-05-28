import numpy as np
from src.oreilly.my.common.myUtil import *

def AND(x1, x2):
    ans = 0
    x = np.array([x1, x2, 1])
    w = 1, 1, -1
    y = np.dot(x, w)
    if(y <= 0):
        ans = 0
    else:
        ans = 1
    return ans

def OR(x1, x2):
    ans = 0
    x = np.array([x1, x2, 1])
    w = 1, 1, 0
    y = np.dot(x, w)
    if(y <= 0):
        ans = 0
    else:
        ans = 1
    return ans

def NAND(x1, x2):
    ans = 0
    x = np.array([x1, x2, 1])
    w = -1, -1, 1.1
    y = np.dot(x, w)
    if(y <= 0):
        ans = 0
    else:
        ans = 1
    return ans

def XOR(x1, x2):
    # 第一層
    s1 = OR(x1, x2)
    s2 = NAND(x1, x2)
    # 第二層
    return AND(s1,s2)

def XOR2(x1, x2):
    x = np.array([x1, x2, 1])
    w1 = np.array([[1, 1, 0],[-1, -1, 1.1], [0, 0, 1]])
    w2 = np.array([1, 1, -1])

    a1 = np.array([step(a) for a in np.dot(w1, x.T)])
    a2 = step(np.dot(w2, a1.T))
    return a2

if(__name__ == '__main__'):
    print("### AND ")
    print("0,0 ->" ,AND(0,0))
    print("1,0 ->" ,AND(1,0))
    print("0,1 ->" ,AND(0,1))
    print("1,1 ->" ,AND(1,1))

    print("### OR ")
    print("0,0 ->" ,OR(0,0))
    print("1,0 ->" ,OR(1,0))
    print("0,1 ->" ,OR(0,1))
    print("1,1 ->" ,OR(1,1))

    print("### NAND ")
    print("0,0 ->" ,NAND(0,0))
    print("1,0 ->" ,NAND(1,0))
    print("0,1 ->" ,NAND(0,1))
    print("1,1 ->" ,NAND(1,1))

    print("### XOR ")
    print("0,0 ->" ,XOR(0,0))
    print("1,0 ->" ,XOR(1,0))
    print("0,1 ->" ,XOR(0,1))
    print("1,1 ->" ,XOR(1,1))

    print("### XOR2 ")
    print("0,0 ->" ,XOR2(0,0))
    print("1,0 ->" ,XOR2(1,0))
    print("0,1 ->" ,XOR2(0,1))
    print("1,1 ->" ,XOR2(1,1))