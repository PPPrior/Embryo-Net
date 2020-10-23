from data import *
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

import matplotlib.pyplot as plt
from skimage.color import rgb2hsv
from skimage.color import rgb2gray
from concurrent.futures import ThreadPoolExecutor


def main():
    root = r'/mnt/data2/like/data/embryo/cropped'
    data_vis(root)


def get_gray(root):
    images, _ = get_data(root)
    pool = ThreadPoolExecutor(max_workers=8)
    pool.map(_get_gray, images)


def _get_gray(image):
    gray = rgb2gray(io.imread(image))
    save_image(gray, image.replace('cropped', 'gray'))


def get_cropped(root):
    images, _ = get_data(root)
    pool = ThreadPoolExecutor(max_workers=8)
    pool.map(_get_cropped, images)


def _get_cropped(image):
    cropped = get_roi(image)
    save_image(cropped, image.replace('medical', 'cropped'))


def data_vis(root):
    ds = EmbryoDataset(root)
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=16)
    iteration = 1
    for img, label in dl:
        img = make_grid(img, nrow=8).numpy()
        img = np.transpose(img, (1, 2, 0))
        plt.figure(figsize=(20, 12))
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


if __name__ == '__main__':
    main()
