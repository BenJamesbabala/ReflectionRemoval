from __future__ import division
import cv2, os
import numpy as np
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

def prepare_data_test(test_path):
    input_names = []
    for dirname in test_path:
        for path, _, fnames in sorted(os.walk(dirname)):
            for fname in fnames:
                if is_image_file(fname):
                    input_names.append(path + '/' + fname)
    # print(input_names)
    return input_names

def prepare_data(train_path):
    input_names = []
    image1 = []
    image2 = []
    for dirname in train_path:
        train_t_gt = dirname + "/transmission_layer/"
        train_r_gt = dirname + "/reflection_layer/"
        train_b = dirname + "/blended/"
        for root, _, fnames in sorted(os.walk(train_t_gt)):
            for fname in fnames:
                if is_image_file(fname):
                    path_input = os.path.join(train_b, fname)
                    path_output1 = os.path.join(train_t_gt, fname)
                    path_output2 = os.path.join(train_r_gt, fname)
                    input_names.append(path_input)
                    image1.append(path_output1)
                    image2.append(path_output2)
    return input_names, image1, image2

# Fetch data from benchmark database
def fetch_data(path):
    if not os.path.exists('./input'):
        os.mkdir('./input')
    if not os.path.exists('./gt'):
        os.mkdir('./gt')
    for _, dirs, _ in os.walk(path):
        for dir in dirs:
            name = '100' + dir + '.jpg'

            m = os.path.join(path, dir)
            m = os.path.join(m, 'm.jpg')

            m1 = os.path.join('input', name)

            g = os.path.join(path, dir)
            g = os.path.join(g, 'g.jpg')
            g1 = os.path.join('gt', name)


            os.rename(m, m1)
            os.rename(g, g1)

# Check if input and ground truth are in pair
def pair_test(dir1, dir2):
    for _, _, files in os.walk(dir1):
        for file in files:
            pair = os.path.join(dir2, file)
            if not os.path.exists(pair):
                print(file, 'Not in pair')
                break;