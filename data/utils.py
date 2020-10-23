import os
import os.path as osp

import glob
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage import io
import numpy as np


def _get_path(rootpth):
    pos = glob.glob(rootpth + '/*' * 2 + '/P*' + '[5-8].jpg')
    neg = glob.glob(rootpth + '/*' * 2 + '/NP*' + '[5-8].jpg')
    return pos, neg


def _get_center(image):
    gray = rgb2gray(image)
    edges = canny(gray, sigma=0)
    hough_radii = np.arange(180, 240, 20)
    hough_res = hough_circle(edges, hough_radii)
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1)
    return cx[0], cy[0]


def _crop(image, x, y, w=256):  # image H x W x C
    height, width, channels = image.shape
    if x - w < 0:
        image = image[:, 0:w * 2, :]
    elif x + w > width:
        image = image[:, width - w * 2:width, :]
    else:
        image = image[:, x - w:x + w, :]
    if y - w < 0:
        image = image[0:w * 2, :, :]
    elif y + w > height:
        image = image[height - w * 2:height, :, :]
    else:
        image = image[y - w:y + w, :, :]

    return image


def get_data(root):
    pos, neg = _get_path(root)
    image = pos + neg
    label = np.append(np.ones(len(pos), dtype=int), np.zeros(len(neg), dtype=int))
    return image, label


def get_roi(path):
    image = io.imread(path)
    x, y = _get_center(image=image)
    image = _crop(image, x, y)

    # image = resize(image, (224, 224, 3))
    # image = np.array(image)[:, :, ::-1] RGB --> BGR
    return image


def save_image(image, path):
    dir = osp.dirname(path)
    if not osp.exists(dir):
        os.makedirs(dir)
    io.imsave(path, image)
