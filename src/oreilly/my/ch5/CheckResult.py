import pickle
from matplotlib import pyplot as plt
import numpy as np

with open('result.pkl', 'rb') as f:
    result = pickle.load(f)

train_loss_list = result['train_loss_list']
train_acc_list = result['train_acc_list']
test_acc_list = result['test_acc_list']

### 損失関数のグラフ
x_list = np.arange(len(train_loss_list))
plt.plot(x_list, train_loss_list)
plt.show()

### 認識精度のグラフ
x_list = np.arange(len(train_acc_list))
plt.plot(x_list, train_acc_list)
plt.plot(x_list, test_acc_list)
plt.show()