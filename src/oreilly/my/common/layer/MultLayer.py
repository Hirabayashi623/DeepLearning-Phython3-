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