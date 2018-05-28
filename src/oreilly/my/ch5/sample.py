import numpy as np
from src.oreilly.my.common.myUtil import *

class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        return x + y

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy

class MultLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        self.x = x
        self.y = y
        return x * y

    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy

class ReluLayer:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        dout = x.copy()
        dout[self.mask] = 0
        return dout

    def backward(self, dout):
        dx = dout.copy()
        dx[self.mask] = 0
        return dx

class SigmoidLayer:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / ( 1 + np.exp(-x) )
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx

class AffineLayer:
    def __init__(self):
        self.w = None
        self.b = None
        self.x = None

    def forward(self, w, x, b):
        self.w = w
        self.b = b
        self.x = x
        out = np.dot(x, w) + b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.w.T)
        dw = np.dot(self.x.T, dout)
        db = np.sum(dout, axis=0)

        return dw, dx, db

class SoftmaxWithCrosslossLayer:
    def __init__(self):
        self.loss = None
        self.y = None # ソフトマックスの出力
        self.t = None # 教師データ

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx

