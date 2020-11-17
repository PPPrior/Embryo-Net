from data import *
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

import matplotlib.pyplot as plt
from skimage.color import rgb2hsv
from skimage.color import rgb2gray
from concurrent.futures import ThreadPoolExecutor
from sklearn import manifold


def main():
    root = r'/mnt/data2/like/data/embryo/cropped'
    data_vis(root, '脱水后')
    # get_cropped(root, '脱水后')

    # X_path, y = get_data(root)
    # X = list()
    # for i, p in enumerate(X_path):
    #     X.append(io.imread(p).reshape(512 ** 2))
    # X = np.array(X)
    # tsne(X, y)


def get_gray(root):
    images, _ = get_data(root)
    pool = ThreadPoolExecutor(max_workers=8)
    pool.map(_get_gray, images)


def _get_gray(image):
    gray = rgb2gray(io.imread(image))
    save_image(gray, image.replace('cropped', 'gray'))


def get_cropped(root, date):
    images, _ = get_data(root, date)
    pool = ThreadPoolExecutor(max_workers=8)
    pool.map(_get_cropped, images)


def _get_cropped(image):
    cropped = get_roi(image)
    save_image(cropped, image.replace('medical', 'cropped'))


def data_vis(root, date):
    ds = EmbryoDataset(*get_data(root, date))
    print(f'Data Length: {len(ds)}')
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=16)
    iteration = 1
    for img, label in dl:
        img = make_grid(img, nrow=8).numpy()
        img = np.transpose(img, (1, 2, 0))
        plt.figure(figsize=(15, 9))
        plt.title('iteration: {}'.format(iteration))
        plt.imshow(img)
        plt.show()

        # showGray(img)
        # splitRGB(img)
        # splitHSV(img)

        iteration += 1


def showGray(image):
    gray = rgb2gray(image)
    plt.figure(figsize=(12, 6))
    plt.imshow(gray, cmap='gray')
    plt.title('gray')
    plt.show()


def splitRGB(image):
    # RGB channels
    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))

    ax1.set_title('Red')
    ax1.imshow(r, cmap='gray')

    ax2.set_title('Green')
    ax2.imshow(g, cmap='gray')

    ax3.set_title('Blue')
    ax3.imshow(b, cmap='gray')

    plt.show()


def splitHSV(image):
    # Convert from RGB to HSV
    hsv = rgb2hsv(image)

    # HSV channels
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))

    ax1.set_title('Hue')
    ax1.imshow(h, cmap='gray')

    ax2.set_title('Saturation')
    ax2.imshow(s, cmap='gray')

    ax3.set_title('Value')
    ax3.imshow(v, cmap='gray')

    plt.show()


def tsne(X, y):
    '''t-SNE'''
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(X)

    '''plot embedding'''
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    plt.figure(figsize=(8, 8))
    for i in range(X_norm.shape[0]):
        plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.show()


if __name__ == '__main__':
    main()
