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