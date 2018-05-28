import numpy as np

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