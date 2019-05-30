from __future__ import division
import os, time, cv2, scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


# functions for synthesizing images with reflection (details in the paper)
def gkern(kernlen=100, nsig=1):
    """Returns a 2D Gaussian kernel array."""
    interval = (2 * nsig + 1.) / (kernlen)
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    kernel = kernel / kernel.max()
    return kernel


# create a vignetting mask
g_mask = gkern(560, 3)
g_mask = np.dstack((g_mask, g_mask, g_mask))


def syn_data(t, r, sigma):
    t = np.power(t, 2.2)
    r = np.power(r, 2.2)

    sz = int(2 * np.ceil(2 * sigma) + 1)
    r_blur = cv2.GaussianBlur(r, (sz, sz), sigma, sigma, 0)
    blend = r_blur + t

    att = 1.08 + np.random.random() / 10.0

    for i in range(3):
        maski = blend[:, :, i] > 1
        mean_i = max(1., np.sum(blend[:, :, i] * maski) / (maski.sum() + 1e-6))
        r_blur[:, :, i] = r_blur[:, :, i] - (mean_i - 1) * att
    r_blur[r_blur >= 1] = 1
    r_blur[r_blur <= 0] = 0

    h, w = r_blur.shape[0:2]
    neww = np.random.randint(0, 560 - w - 10)
    newh = np.random.randint(0, 560 - h - 10)
    alpha1 = g_mask[newh:newh + h, neww:neww + w, :]
    alpha2 = 1 - np.random.random() / 5.0;
    r_blur_mask = np.multiply(r_blur, alpha1)
    blend = r_blur_mask + t * alpha2

    t = np.power(t, 1 / 2.2)
    r_blur_mask = np.power(r_blur_mask, 1 / 2.2)
    blend = np.power(blend, 1 / 2.2)
    blend[blend >= 1] = 1
    blend[blend <= 0] = 0

    return t, r_blur_mask, blend
