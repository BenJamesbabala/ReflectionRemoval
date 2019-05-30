from __future__ import division
import os, time, cv2, scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import argparse

from discriminator import build_discriminator
from model import build, compute_l1_loss, compute_percep_loss, compute_exclusion_loss, compute_gradient
from utils import is_image_file, gkern, syn_data

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="pre-trained", help="path to folder containing the model")
parser.add_argument("--data_syn_dir", default="", help="path to synthetic dataset")
parser.add_argument("--data_real_dir", default="", help="path to real dataset")
parser.add_argument("--save_model_freq", default=1, type=int, help="frequency to save model")
parser.add_argument("--is_hyper", default=1, type=int, help="use hypercolumn or not")
parser.add_argument("--is_training", default=1, help="training or testing")
parser.add_argument("--continue_training", action="store_true",
                    help="search for checkpoint in the subfolder specified by `task` argument")
ARGS = parser.parse_args()

task = ARGS.task
is_training = ARGS.is_training == 1
continue_training = ARGS.continue_training
hyper = ARGS.is_hyper == 1

# os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
if is_training:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
EPS = 1e-12
channel = 64  # number of feature channels to build the model, set to 64

train_syn_root = [ARGS.data_syn_dir]
train_real_root = [ARGS.data_real_dir]

# set up the model and define the graph
with tf.variable_scope(tf.get_variable_scope()):
    input = tf.placeholder(tf.float32, shape=[None, None, None, 3])
    target = tf.placeholder(tf.float32, shape=[None, None, None, 3])
    reflection = tf.placeholder(tf.float32, shape=[None, None, None, 3])
    issyn = tf.placeholder(tf.bool, shape=[])

    # build the model
    network = build(input)
    transmission_layer, reflection_layer = tf.split(network, num_or_size_splits=2, axis=3)

    # Perceptual Loss
    loss_percep_t = compute_percep_loss(transmission_layer, target)
    loss_percep_r = tf.where(issyn, compute_percep_loss(reflection_layer, reflection, reuse=True), 0.)
    loss_percep = tf.where(issyn, loss_percep_t + loss_percep_r, loss_percep_t)

    # Adversarial Loss
    with tf.variable_scope("discriminator"):
        predict_real, pred_real_dict = build_discriminator(input, target)
    with tf.variable_scope("discriminator", reuse=True):
        predict_fake, pred_fake_dict = build_discriminator(input, transmission_layer)

    d_loss = (tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))) * 0.5
    g_loss = tf.reduce_mean(-tf.log(predict_fake + EPS))

    # L1 loss on reflection image
    loss_l1_r = tf.where(issyn, compute_l1_loss(reflection_layer, reflection), 0)

    # Gradient loss
    loss_gradx, loss_grady = compute_exclusion_loss(transmission_layer, reflection_layer, level=3)
    loss_gradxy = tf.reduce_sum(sum(loss_gradx) / 3.) + tf.reduce_sum(sum(loss_grady) / 3.)
    loss_grad = tf.where(issyn, loss_gradxy / 2.0, 0)

    loss = loss_l1_r + loss_percep * 0.2 + loss_grad

train_vars = tf.trainable_variables()
d_vars = [var for var in train_vars if 'discriminator' in var.name]
g_vars = [var for var in train_vars if 'g_' in var.name]
g_opt = tf.train.AdamOptimizer(learning_rate=0.0002).minimize(loss * 100 + g_loss, var_list=g_vars)  # optimizer for the generator
d_opt = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(d_loss, var_list=d_vars)  # optimizer for the discriminator

for var in tf.trainable_variables():
    print("Listing trainable variables ... ")
    print(var)

saver = tf.train.Saver(max_to_keep=10)

######### Session #########
sess = tf.Session()
sess.run(tf.global_variables_initializer())
ckpt = tf.train.get_checkpoint_state(task)
print("[i] contain checkpoint: ", ckpt)
if ckpt and continue_training:
    saver_restore = tf.train.Saver([var for var in tf.trainable_variables()])
    print('loaded ' + ckpt.model_checkpoint_path)
    saver_restore.restore(sess, ckpt.model_checkpoint_path)
# test doesn't need to load discriminator
else:
    saver_restore = tf.train.Saver([var for var in tf.trainable_variables() if 'discriminator' not in var.name])
    print('loaded ' + ckpt.model_checkpoint_path)
    saver_restore.restore(sess, ckpt.model_checkpoint_path)

maxepoch = 100
k_sz = np.linspace(1, 5, 80)  # for synthetic images
if is_training:
    # please follow the dataset directory setup in README
    def prepare_data(train_path):
        input_names = []
        image1 = []
        image2 = []
        for dirname in train_path:
            train_t_gt = dirname + "transmission_layer/"
            train_r_gt = dirname + "reflection_layer/"
            train_b = dirname + "blended/"
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


    _, syn_image1_list, syn_image2_list = prepare_data(
        train_syn_root)  # image pairs for generating synthetic training images
    input_real_names, output_real_names1, output_real_names2 = prepare_data(
        train_real_root)  # no reflection ground truth for real images
    print("[i] Total %d training images, first path of real image is %s." % (
    len(syn_image1_list) + len(output_real_names1), input_real_names[0]))

    num_train = len(syn_image1_list) + len(output_real_names1)
    all_l = np.zeros(num_train, dtype=float)
    all_percep = np.zeros(num_train, dtype=float)
    all_grad = np.zeros(num_train, dtype=float)
    all_g = np.zeros(num_train, dtype=float)
    for epoch in range(1, maxepoch):
        input_images = [None] * num_train
        output_images_t = [None] * num_train
        output_images_r = [None] * num_train

        if os.path.isdir("%s/%04d" % (task, epoch)):
            continue
        cnt = 0
        for id in np.random.permutation(num_train):
            st = time.time()
            if input_images[id] is None:
                magic = np.random.random()
                if magic < 0.7:  # choose from synthetic dataset
                    is_syn = True
                    syn_image1 = cv2.imread(syn_image1_list[id], -1)
                    neww = np.random.randint(256, 480)
                    newh = round((neww / syn_image1.shape[1]) * syn_image1.shape[0])
                    output_image_t = cv2.resize(np.float32(syn_image1), (neww, newh), cv2.INTER_CUBIC) / 255.0
                    output_image_r = cv2.resize(np.float32(cv2.imread(syn_image2_list[id], -1)), (neww, newh),
                                                cv2.INTER_CUBIC) / 255.0
                    file = os.path.splitext(os.path.basename(syn_image1_list[id]))[0]
                    sigma = k_sz[np.random.randint(0, len(k_sz))]
                    if np.mean(output_image_t) * 1 / 2 > np.mean(output_image_r):
                        continue
                    _, output_image_r, input_image = syn_data(output_image_t, output_image_r, sigma)
                else:  # choose from real dataste
                    is_syn = False
                    _id = id % len(input_real_names)
                    inputimg = cv2.imread(input_real_names[_id], -1)
                    file = os.path.splitext(os.path.basename(input_real_names[_id]))[0]
                    neww = np.random.randint(256, 480)
                    newh = round((neww / inputimg.shape[1]) * inputimg.shape[0])
                    input_image = cv2.resize(np.float32(inputimg), (neww, newh), cv2.INTER_CUBIC) / 255.0
                    output_image_t = cv2.resize(np.float32(cv2.imread(output_real_names1[_id], -1)), (neww, newh),
                                                cv2.INTER_CUBIC) / 255.0
                    output_image_r = output_image_t  # reflection gt not necessary
                    sigma = 0.0
                input_images[id] = np.expand_dims(input_image, axis=0)
                output_images_t[id] = np.expand_dims(output_image_t, axis=0)
                output_images_r[id] = np.expand_dims(output_image_r, axis=0)

                # remove some degenerated images (low-light or over-saturated images), heuristically set
                if output_images_r[id].max() < 0.15 or output_images_t[id].max() < 0.15:
                    print("Invalid reflection file %s (degenerate channel)" % (file))
                    continue
                if input_images[id].max() < 0.1:
                    print("Invalid file %s (degenerate image)" % (file))
                    continue

                # alternate training, update discriminator every two iterations
                if cnt % 2 == 0:
                    fetch_list = [d_opt]
                    # update D
                    _ = sess.run(fetch_list, feed_dict={input: input_images[id], target: output_images_t[id]})
                fetch_list = [g_opt, transmission_layer, reflection_layer,
                              d_loss, g_loss,
                              loss, loss_percep, loss_grad]
                # update G
                _, pred_image_t, pred_image_r, current_d, current_g, current, current_percep, current_grad = \
                    sess.run(fetch_list, feed_dict={input: input_images[id], target: output_images_t[id],
                                                    reflection: output_images_r[id], issyn: is_syn})

                all_l[id] = current
                all_percep[id] = current_percep
                all_grad[id] = current_grad * 255
                all_g[id] = current_g
                g_mean = np.mean(all_g[np.where(all_g)])
                print(
                            "iter: %d %d || D: %.2f || G: %.2f %.2f || all: %.2f || loss: %.2f %.2f || mean: %.2f %.2f || time: %.2f" %
                            (epoch, cnt, current_d, current_g, g_mean,
                             np.mean(all_l[np.where(all_l)]),
                             current_percep, current_grad * 255,
                             np.mean(all_percep[np.where(all_percep)]), np.mean(all_grad[np.where(all_grad)]),
                             time.time() - st))
                cnt += 1
                input_images[id] = 1.
                output_images_t[id] = 1.
                output_images_r[id] = 1.

        # save model and images every epoch
        if epoch % ARGS.save_model_freq == 0:
            os.makedirs("%s/%04d" % (task, epoch))
            saver.save(sess, "%s/model.ckpt" % task)
            saver.save(sess, "%s/%04d/model.ckpt" % (task, epoch))

            fileid = os.path.splitext(os.path.basename(syn_image1_list[id]))[0]
            if not os.path.isdir("%s/%04d/%s" % (task, epoch, fileid)):
                os.makedirs("%s/%04d/%s" % (task, epoch, fileid))
            pred_image_t = np.minimum(np.maximum(pred_image_t, 0.0), 1.0) * 255.0
            pred_image_r = np.minimum(np.maximum(pred_image_r, 0.0), 1.0) * 255.0
            print("shape of outputs: ", pred_image_t.shape, pred_image_r.shape)
            cv2.imwrite("%s/%04d/%s/int_t.png" % (task, epoch, fileid), np.uint8(np.squeeze(input_image * 255.0)))
            cv2.imwrite("%s/%04d/%s/out_t.png" % (task, epoch, fileid), np.uint8(np.squeeze(pred_image_t)))
            cv2.imwrite("%s/%04d/%s/out_r.png" % (task, epoch, fileid), np.uint8(np.squeeze(pred_image_r)))
# To test the model on images with reflection
else:
    def prepare_data_test(test_path):
        input_names = []
        for dirname in test_path:
            for _, _, fnames in sorted(os.walk(dirname)):
                for fname in fnames:
                    if is_image_file(fname):
                        input_names.append(os.path.join(dirname, fname))
        return input_names


    # Please replace with your own test image path
    test_path = ["./test_images/CEILNet/"]  # ["./test_images/real/"]
    subtask = "CEILNet"  # if you want to save different testset separately
    val_names = prepare_data_test(test_path)

    for val_path in val_names:
        testind = os.path.splitext(os.path.basename(val_path))[0]
        if not os.path.isfile(val_path):
            continue
        img = cv2.imread(val_path)
        input_image = np.expand_dims(np.float32(img), axis=0) / 255.0
        st = time.time()
        output_image_t, output_image_r = sess.run([transmission_layer, reflection_layer],
                                                  feed_dict={input: input_image})
        print("Test time %.3f for image %s" % (time.time() - st, val_path))
        output_image_t = np.minimum(np.maximum(output_image_t, 0.0), 1.0) * 255.0
        output_image_r = np.minimum(np.maximum(output_image_r, 0.0), 1.0) * 255.0
        if not os.path.isdir("./test_results/%s/%s" % (subtask, testind)):
            os.makedirs("./test_results/%s/%s" % (subtask, testind))
        cv2.imwrite("./test_results/%s/%s/input.png" % (subtask, testind), img)
        cv2.imwrite("./test_results/%s/%s/t_output.png" % (subtask, testind),
                    np.uint8(output_image_t[0, :, :, 0:3]))  # output transmission layer
        cv2.imwrite("./test_results/%s/%s/r_output.png" % (subtask, testind),
                    np.uint8(output_image_r[0, :, :, 0:3]))  # output reflection layer

