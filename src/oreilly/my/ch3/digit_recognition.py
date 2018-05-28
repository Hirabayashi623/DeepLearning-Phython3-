from src.oreilly.my.common.mnist import *

from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

if __name__ == '__main__':
    # 訓練データとテストデータを読込
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

    img = x_train[0]
    label = t_train[0]
    print(label)

    # 画像データ
    img = img.reshape(28, 28)
    img_show(img)
