import numpy as np
from matplotlib  import pyplot as plt
from src.oreilly.my.ch5.TwoLayerNet import TwoLayerNet
from src.oreilly.my.common.mnist import *
import pickle

# データの読込
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# ニューラルネットワークのインスタンス化
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

###################
## 各種設定値
###################
iterate_times = 100
iterate_per_epoch = 1
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1


train_loss_list = []
train_acc_list = []
test_acc_list = []

for i in np.arange(iterate_times):
    # 訓練データの要素番号の中からランダムでバッチ数分取得する
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 勾配を取得
    grad = network.gradient(x_batch, t_batch)

    # パラメータの更新
    for key in ('w1', 'b1', 'w2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # 損失関数を計算
    loss = network.loss(x_batch, t_batch)
    # グラフ用データに損失関数を追加
    train_loss_list.append(loss)

    # コンソール出力
    print(i,'回目: ', loss)

    if i % iterate_per_epoch == 0:
        # 認識精度を取得
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        # グラフ用リストに追加
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)


# 結果をファイルに保存
result = {}
result['train_loss_list'] = train_loss_list
result['train_acc_list'] = train_acc_list
result['test_acc_list'] = test_acc_list
result['params'] = network.params

with open('result.pkl', 'wb') as f:
    pickle.dump(result, f, -1)

