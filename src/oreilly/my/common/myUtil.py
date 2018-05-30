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

def softmax(x):
    # 2次元データの場合
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)   # オーバーフロー対策
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x) # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x))

if __name__ == '__main__':
    print(step(100))


## 損失関数
# 平均二乗誤差
def mean_square_error(y, t):
    return 0.5 * np.sum((y - t)**2)

# 交差エントロピー誤差
def cross_entropy_error(y, t):
    delta = 1e-7
#     print('### cross_entropy_error')
#     print('y: ', y)
#     print('t: ', t)
    return -np.sum(np.log(y) * t + delta)

### 勾配法の実装
def gradient_descent(gradient, x, lr):
    return x - gradient(x) * lr

## 数値微分によるパラメータ更新
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

