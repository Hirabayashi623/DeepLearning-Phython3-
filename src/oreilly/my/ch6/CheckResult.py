import pickle
from matplotlib import pyplot as plt
import numpy as np

with open('result.pkl', 'rb') as f:
    result = pickle.load(f)

loss_list = result['loss']
x_list = result['x']
y_list = result['y']

mode = 0

def loss_graf():
    for key in loss_list.keys():
        loss = loss_list[key]
        plt.scatter(np.arange(len(loss)), loss, label=key, s=10)

    plt.legend()
    plt.show()

def track():
    for key in x_list.keys():
        plt.scatter(x_list[key], y_list[key], label=key, s=10)

    plt.legend()
    plt.show()


if mode == 0:
    track()
elif mode ==1:
    loss_graf()
else:
    print('unkwonw')