from __future__ import division
import os, time, cv2
import tensorflow as tf
import numpy as np
import argparse

from discriminator import build_discriminator
from model import build, compute_percep_loss, compute_l1_loss, compute_exclusion_loss
from utils import prepare_data

from tensorboard_logging import Logger

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="pre-trained", help="path to folder containing the model")
parser.add_argument("--data_real_dir", default="root_training_real_data", help="path to real dataset")
parser.add_argument("--save_model_freq", default=1, type=int, help="frequency to save model")
parser.add_argument("--is_hyper", default=1, type=int, help="use hypercolumn or not")
parser.add_argument("--continue_training", action="store_true",
                    help="search for checkpoint in the subfolder specified by `task` argument")
ARGS = parser.parse_args()

task = ARGS.task
continue_training = ARGS.continue_training
hyper = ARGS.is_hyper == 1

maxepoch = 300
# os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
EPS = 1e-12
channel = 64  # number of feature channels to build the model, set to 64

train_real_root = [ARGS.data_real_dir]

# set up the model and define the graph
with tf.variable_scope(tf.get_variable_scope()):
    input = tf.placeholder(tf.float32, shape=[None, None, None, 3])
    target = tf.placeholder(tf.float32, shape=[None, None, None, 3])
    reflection = tf.placeholder(tf.float32, shape=[None, None, None, 3])

    # build the model
    network = build(input, hyper, channel)
    transmission_layer, reflection_layer = tf.split(network, num_or_size_splits=2, axis=3)

    # Perceptual Loss
    loss_percep_t = compute_percep_loss(transmission_layer, target)
    loss_percep_r = compute_percep_loss(reflection_layer, reflection, reuse=True)
    loss_percep = loss_percep_t + loss_percep_r

    # L1 loss on reflection image
    loss_l1_r = compute_l1_loss(reflection_layer, reflection)

    # Adversarial Loss
    with tf.variable_scope("discriminator"):
        predict_real, pred_real_dict = build_discriminator(input, target)
    with tf.variable_scope("discriminator", reuse=True):
        predict_fake, pred_fake_dict = build_discriminator(input, transmission_layer)

    d_loss = (tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))) * 0.5
    g_loss = tf.reduce_mean(-tf.log(predict_fake + EPS))

    # Gradient loss
    loss_gradx, loss_grady = compute_exclusion_loss(transmission_layer, reflection_layer, level=3)
    loss_grad = tf.reduce_sum(sum(loss_gradx) / 3.) + tf.reduce_sum(sum(loss_grady) / 3.)

    loss = loss_l1_r * 0.5 + loss_percep * 0.2 + loss_grad * 0.1 + g_loss * 0.01

train_vars = tf.trainable_variables()
d_vars = [var for var in train_vars if 'discriminator' in var.name]
g_vars = [var for var in train_vars if 'g_' in var.name]
g_opt = tf.train.AdamOptimizer(learning_rate=0.002).minimize(loss, var_list=g_vars)  # optimizer for the generator
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
    print('continue_training', continue_training)
    saver_restore = tf.train.Saver([var for var in tf.trainable_variables()])
    print('loaded ' + ckpt.model_checkpoint_path)
    saver_restore.restore(sess, ckpt.model_checkpoint_path)

input_real_names, output_real_names1, _ = prepare_data(train_real_root)  # no reflection ground truth for real images
print("[i] Total %d training images, first path of real image is %s." % (len(output_real_names1), input_real_names[0]))

num_train = len(output_real_names1)
logger = Logger('logs')

for epoch in range(1, maxepoch):
    sum_p = 0
    sum_g = 0
    sum_d = 0
    sum_grad = 0
    sum_loss = 0

    sum_mse = 0
    sum_psnr = 0

    input_images = [None] * num_train
    output_images_t = [None] * num_train
    output_images_r = [None] * num_train

    if os.path.isdir("%s/%04d" % (task, epoch)):
        continue
    cnt = 0
    for id in np.random.permutation(num_train):
        st = time.time()
        if input_images[id] is None:
            _id = id % len(input_real_names)
            inputimg = cv2.imread(input_real_names[_id], -1)
            file = os.path.splitext(os.path.basename(input_real_names[_id]))[0]
            neww = np.random.randint(256, 480)
            newh = round((neww / inputimg.shape[1]) * inputimg.shape[0])
            input_image = cv2.resize(np.float32(inputimg), (neww, newh), cv2.INTER_CUBIC) / 255.0
            output_image_t = cv2.resize(np.float32(cv2.imread(output_real_names1[_id], -1)), (neww, newh),
                                        cv2.INTER_CUBIC) / 255.0
            sigma = 0.0
            input_images[id] = np.expand_dims(input_image, axis=0)
            output_images_t[id] = np.expand_dims(output_image_t, axis=0)

            # remove some degenerated images (low-light or over-saturated images), heuristically set
            if output_images_t[id].max() < 0.15:
                print("Invalid reflection file %s (degenerate channel)" % (file))
                continue
            if input_images[id].max() < 0.1:
                print("Invalid file %s (degenerate image)" % (file))
                continue

            # alternate training, update discriminator every two iterations
            if cnt % 2 == 0:
                # update D
                fetch_list = [d_opt]
                _ = sess.run(fetch_list, feed_dict={input: input_images[id], target: output_images_t[id]})
            # update G
            fetch_list = [g_opt,
                          transmission_layer, reflection_layer,
                          d_loss, g_loss,
                          loss_percep, loss_grad, loss]
            _, pred_image_t, pred_image_r, current_d, current_g, current_p, current_grad, current_loss= \
                sess.run(fetch_list, feed_dict={input: input_images[id], target: output_images_t[id]})

            sum_p += current_p
            sum_g += current_g
            sum_d += current_d
            sum_grad += current_grad
            sum_loss += current_loss

            print("iter: %d %d || D: %.2f || G: %.2f || P: %.2f || GRAD: %.2f || ALL: %.2f || time: %.2f" %
                        (epoch, cnt,
                         current_d,
                         current_g,
                         current_p,
                         current_grad,
                         current_loss,
                         time.time() - st))
            cnt += 1
            input_images[id] = 1.
            output_images_t[id] = 1.
            output_images_r[id] = 1.

    test_path = ['dev_images/']
    b, t, _ = prepare_data(test_path)
    n = len(b)
    for i in range(n):
        img = cv2.imread(b[i])
        input_image = np.expand_dims(np.float32(img), axis=0) / 255.0
        img1 = cv2.imread(t[i])
        input_image_t = np.expand_dims(np.float32(img1), axis=0) / 255.0

        output_image_t = sess.run([transmission_layer], feed_dict={input: input_image})
        output_image_t = np.minimum(np.maximum(output_image_t, 0.0), 1.0) * 255.0
        print(input_image_t.shape, output_image_t)

        mse = ((input_image_t - output_image_t) ** 2).mean()
        sum_mse += mse
        sum_psnr += 10. * np.log10(1. / mse)

    sum_p /= cnt
    sum_g /= cnt
    sum_d /= cnt
    sum_grad /= cnt
    sum_loss /= cnt
    sum_mse /= n
    sum_psnr /= n

    # print('==========', sum_p, sum_g, sum_d)
    logger.log_scalar('generator loss', sum_g, epoch)
    logger.log_scalar('perceptual loss', sum_p, epoch)
    logger.log_scalar('discriminator loss', sum_d, epoch)
    logger.log_scalar('gradient loss', sum_grad, epoch)
    logger.log_scalar('total loss', sum_loss, epoch)
    logger.log_scalar('MSE', sum_mse, epoch)
    logger.log_scalar('PSNR', sum_psnr, epoch)

    # save model and images every epoch
    if epoch > 20 and epoch % ARGS.save_model_freq == 0:
        os.makedirs("%s/%04d" % (task, epoch))
        saver.save(sess, "%s/model.ckpt" % task)
        saver.save(sess, "%s/%04d/model.ckpt" % (task, epoch))




