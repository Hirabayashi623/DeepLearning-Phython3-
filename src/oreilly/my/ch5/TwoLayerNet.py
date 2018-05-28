from src.oreilly.my.common.layer.AffineLayer import AffineLayer
from src.oreilly.my.common.layer.SigmoidLayer import SigmoidLayer
from src.oreilly.my.common.layer.SoftmaxWithCrosslossLayer import SoftmaxWithCrosslossLayer
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

        # Layerの生成
        self.layer = {}
        self.layer['affine'] = AffineLayer()
        self.layer['sigmoid'] = SigmoidLayer()
        self.layer['softmax'] = SoftmaxWithCrosslossLayer()

    # 推論メソッド
    def predict(self, x):
        w1, w2 = self.params['w1'], self.params['w2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = self.layer['affine'].forward(w1, x, b1)
        z1 = self.layer['sigmoid'].forward(x=a1)
        a2 = self.layer['affine'].forward(w=w2, x=z1, b=b2)
        z2 = a2 # ソフトマックス関数の使用は別で実装するためそのまま

        return z2

    # 訓練データと教師データを与えて損失関数をもとめる
    def loss(self, x, t):
        y = self.predict(x)
        return self.layer['softmax'].forward(x=y, t=t)

    # 認識精度を求める
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)

        return np.sum(y == t) / float(x.shape[0])

    # 勾配を求める
    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # ソフトマックス関数のバックワード
        dout = self.layer['softmax'].backward(1)
        # 第二層のバックワード
        dw2, dx2, db2 = self.layer['affine'].backward(dout)
        # シグモイド関数のバックワード
        dout = self.layer['sigmoid'].backward(dout=dx2)
        # 第一層のバックワード
        dw1, dx1, db1 = self.layer['affine'].backward(dout)

        grads = {}
        grads['w1'] = dw1
        grads['w2'] = dw2
        grads['b1'] = db1
        grads['b2'] = db2

        return grads
