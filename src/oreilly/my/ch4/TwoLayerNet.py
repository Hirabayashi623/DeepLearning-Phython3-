import numpy as np
from src.oreilly.my.common.myUtil import *

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 重みとバイアスの定義
        self.params = {}
        self.params['w1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['w2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    # 推論メソッド
    def predict(self, x):
        w1, w2 = self.params['w1'], self.params['w2']
        b1, b2 = self.params['b1'], self.params['b2']

        # 入力に対する信号を求める（第一層）
        a1 = np.dot(x, w1) + b1
        # 活性化関数で平仄を合わせる（０～１）
        z1 = sigmoid(a1)
        # 入力に対する信号を求める（第二層）
        a2 = np.dot(z1, w2) + b2
        # ソフトマックス関数で正解確率に変換
        z2 = softmax(a2)

        return z2

    # 訓練データと教師データを与えて損失関数をもとめる
    def loss(self, x, t):
        y = self.predict(x)

        return cross_entropy_error(y, t)

    # 認識精度を求める
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
#         t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # 勾配を求める
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['w1'] = numeric_gradient(loss_W, self.params['w1'], 0.1)
        grads['w2'] = numeric_gradient(loss_W, self.params['w2'], 0.1)
        grads['b1'] = numeric_gradient(loss_W, self.params['b1'], 0.1)
        grads['b2'] = numeric_gradient(loss_W, self.params['b2'], 0.1)

        return grads
