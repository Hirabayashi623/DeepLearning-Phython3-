import numpy as np
from matplotlib import pyplot as plt
from src.oreilly.my.ch6.optimizer import *
from src.oreilly.my.common.myUtil import *
import pickle

def f(x):
    return x[0]**2 / 20 + x[1]**2

def f_grad(p):
    grad = {}
    grad['x'] = p['x'] * 0.1
    grad['y'] = p['y'] * 2
    return grad

## 初期設定
x_list = {}
y_list = {}
loss_list = {}
p = {}
p['x'] = 3
p['y'] = -4
learning_rate = 0.01
iterate_times = 1000
# 正解（最小値）
t = np.array([0.0, 0.0])

# オプティマイザの選択
key, optimizer = 'SGD', SGD(lr=learning_rate)
loss_list[key] = []
x_list[key] = []
y_list[key] = []
# 初期位置
p['x'] = 3
p['y'] = -4
for i in np.arange(iterate_times):
    # パラメータの更新
    optimizer.update(p, f_grad(p))
    # グラフ用のデータ追加
    x_list[key].append(p['x'])
    y_list[key].append(p['y'])
    # 損失関数
    x = np.array([p['x'], p['y']])
    loss_list[key].append(mean_square_error(x, t))

# オプティマイザの選択
key, optimizer = 'Momentum', Momentum(lr=learning_rate)
loss_list[key] = []
x_list[key] = []
y_list[key] = []
# 初期位置
p['x'] = 3
p['y'] = -4
for i in np.arange(iterate_times):
    # パラメータの更新
    optimizer.update(p, f_grad(p))
    # グラフ用のデータ追加
    x_list[key].append(p['x'])
    y_list[key].append(p['y'])
    # 損失関数
    x = np.array([p['x'], p['y']])
    loss_list[key].append(mean_square_error(x, t))

# オプティマイザの選択
key, optimizer = 'AdaGrad', AdaGrad(lr=learning_rate)
loss_list[key] = []
x_list[key] = []
y_list[key] = []
# 初期位置
p['x'] = 3
p['y'] = -4
for i in np.arange(iterate_times):
    # パラメータの更新
    optimizer.update(p, f_grad(p))
    # グラフ用のデータ追加
    x_list[key].append(p['x'])
    y_list[key].append(p['y'])
    # 損失関数
    x = np.array([p['x'], p['y']])
    loss_list[key].append(mean_square_error(x, t))

### ファイル保存処理
result = {}
result['loss'] = loss_list
result['x'] = x_list
result['y'] = y_list

with open('result.pkl', 'wb') as f:
    pickle.dump(result, f, -1)

print('finished')
