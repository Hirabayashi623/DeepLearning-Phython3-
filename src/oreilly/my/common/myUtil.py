from matplotlib import pyplot as plt
import numpy as np
import logging


def autoplot(func, xrange1=-5, xrange2=5, sample=10000):
    xlist = np.arange(xrange1, xrange2, (xrange2-xrange1)/sample)
    ylist = np.array([func(x) for x in xlist])
    plt.plot(xlist, ylist)
    plt.show()

### 諸関数
## 活性化関数
def sigmoid(x):
    return 1 / ( 1 + np.exp(-x) )

def step(x):
    if x > 0:
        return 1
    else:
        return 0

def ReLU(x):
    return np.max(0, x)

def softmax(xlist):
    # a / b
    max = np.max(xlist)
    a = np.array([np.exp(xi-max) for xi in xlist])
    b = np.sum(a)
    return a / b;

if __name__ == '__main__':
    print(step(100))


## 損失関数
# 平均二乗誤差
def mean_square_error(y, t):
    return 0.5 * np.sum((y - t)**2)

# 交差エントロピー誤差
def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

### 勾配法の実装
def gradient_descent(gradient, x, lr):
    return x - gradient(x) * lr

## 数値微分によるパラメータ更新
# w⇒パラメータ(更新対象)
# x⇒位置
def numeric_gradient_learn(f, w, x, lr=0.1):
    return 1

def numeric_gradient(f, x, lr=1.0):
    delta = 1e-4
    delta_ = 1e4

    if x.ndim == 1:
        grad = np.arange(x.size,dtype=np.float32)
        for idx in np.arange(x.size):
            x_tmp = x[idx]
            x[idx] = x_tmp + delta
            fxh1 = f(x) # f(x+dx)
            x[idx] = x_tmp - delta
            fxh2 = f(x)
            grad[idx] =  0.5 * ( fxh1 - fxh2 ) * delta_
            x[idx] = x_tmp
    else:
        grad = np.zeros((x.shape[0], x.shape[1]))
        for i, x_ in enumerate(x):
            for j in np.arange(x_.size):
                x_tmp = x_[j]
                x_[j] = x_tmp + delta
                fxh1 = f(x_) # f(x+dx)
                x_[j] = x_tmp - delta
                fxh2 = f(x_)
                grad[i][j] =  0.5 * ( fxh1 - fxh2 ) * delta_
                x_[j] = x_tmp
    return lr * grad

