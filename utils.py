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

def fetch_test_data(path):
    if not os.path.exists('./input'):
        os.mkdir('./input')
    for _, dirs, _ in os.walk(path):
        for dir in dirs:
            name = dir + '.jpg'
            m = os.path.join(path, dir)
            m = os.path.join(m, 'm.jpg')
            m1 = os.path.join('input', name)
            os.rename(m, m1)

# Check if input and ground truth are in pair
def pair_test(dir1, dir2):
    for _, _, files in os.walk(dir1):
        for file in files:
            pair = os.path.join(dir2, file)
            if not os.path.exists(pair):
                print(file, 'Not in pair')
                break;

def normalize(input, mean):
    assert (input.shape[2] == 3 and len(mean) == 3)

    input[:, :, 0] -= mean[0]
    input[:, :, 1] -= mean[1]
    input[:, :, 2] -= mean[2]

    return input


def denormalize(input, mean):
    assert (input.shape[2] == 3 and len(mean) == 3)

    input[:, :, 0] += mean[0]
    input[:, :, 1] += mean[1]
    input[:, :, 2] += mean[2]

    return input