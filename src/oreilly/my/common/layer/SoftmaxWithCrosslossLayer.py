from src.oreilly.my.common.myUtil import *

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
        dx = dout * (self.y - self.t) / batch_size

        return dx