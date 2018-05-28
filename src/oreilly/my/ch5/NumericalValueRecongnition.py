from src.oreilly.my.common.layer import *
from src.oreilly.my.common.mnist import *
import numpy as np
from matplotlib import pyplot as plt
from src.oreilly.my.ch5.TwoLayerNet import TwoLayerNet
from src.oreilly.my.common.myUtil import softmax


if __name__ == '__main__':
    # 訓練データとテストデータの取得
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

    # 訓練データの数を取得
    train_size = x_train.shape[0]
    # バッチサイズを指定
    batch_size = 10
    # 訓練データ抽出用にマスキング配列を生成
    batch_mask = np.random.choice(train_size, batch_size)

    # ミニバッチ用のデータを取得
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # ニューラルネットワークを生成
    net = TwoLayerNet(input_size=784, hidden_size=30, output_size=10)

    # 推論
    ans = softmax(net.predict(x_batch))
    print('### 推論 ###\n', np.sum(ans, axis=1),'\n', np.argmax(ans, axis=1))

    # 認識精度
    ans = net.accuracy(x_batch, t_batch)
    print('### 認識精度 ###\n', ans)

    # 教師データ
    print('### 教師データ ###\n', t_batch)