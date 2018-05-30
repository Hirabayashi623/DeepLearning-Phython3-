import numpy as np
from collections import OrderedDict
from src.oreilly.my.common.layer.layers import *

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 重みの初期化
        self.params = {}
        self.params['w1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['w2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['b2'] = np.zeros(output_size)

        # レイヤの生成
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['w1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['w2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()

    # 推論
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    # 損失関数を求める
    def loss(self, x, t):
        # 推論
        y = self.predict(x)
        # ソフトマックス関数を用いて確率に変換
        # 損失関数を求める
        return self.lastLayer.forward(y, t)

    # 認識精度
    def accuracy(self, x, t):
        # 推論
        y = self.predict(x)
        # テストデータごとの確率が一番高い要素番号（推論ラベル）を取得
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        # 認識精度の計算
        accuracy = np.sum(y == t) / x.shape[0]

        return accuracy

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        # レイヤをコピー
        layers = list(self.layers.values())
        # バックワード処理のために逆順にする
        layers.reverse()
        # バックワード
        for layer in layers:
            dout = layer.backward(dout)

        # 勾配を取得
        grads = {}
        grads['w1'], grads['b1'] = self.layers['Affine1'].dw, self.layers['Affine1'].db
        grads['w2'], grads['b2'] = self.layers['Affine2'].dw, self.layers['Affine2'].db

        return grads