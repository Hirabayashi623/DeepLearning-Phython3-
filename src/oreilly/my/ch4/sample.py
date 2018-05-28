from src.oreilly.my.common.mnist import *
import numpy as np
from matplotlib import pyplot as plt
from src.oreilly.my.ch4.TwoLayerNet import TwoLayerNet

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

    print(x_batch.shape)

    # ニューラルネットワークを生成
    net = TwoLayerNet(input_size=784, hidden_size=30, output_size=10)

    # 推論
    ans = net.predict(x_batch)
    print('### 推論 ###\n', np.sum(ans, axis=1), np.argmax(ans, axis=1))

    # 認識精度
    ans = net.accuracy(x_batch, t_batch)
    print('### 認識精度 ###\n', ans)

    # 認識精度のグラフ用データ
    acc_train = []
    acc_test = []

    for i in np.arange(10):
        # 訓練データの数を取得
        train_size = x_train.shape[0]
        # バッチサイズを指定
        batch_size = 10
        # 訓練データ抽出用にマスキング配列を生成
        batch_mask = np.random.choice(train_size, batch_size)

        # ミニバッチ用のデータを取得
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        # 勾配を取得
        grad = net.numerical_gradient(x_batch, t_batch)
        print('### 勾配 ###\n', grad)

        # パラメータの更新
        for key in ('w1', 'w2', 'b1', 'b2'):
            net.params[key] -= grad[key]

        # 認識精度を取得
        acc_train.append(net.accuracy(x_batch, t_batch))
    x = np.arange(len(acc_train))
    plt.plot(x, acc_train)
    plt.show()