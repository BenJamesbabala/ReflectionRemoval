from utils import prepare_data_test
import cv2, time, os
import numpy as np
import tensorflow as tf
from model import build
import argparse
from math import log10

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="pre-trained", help="path to folder containing the model")
ARGS = parser.parse_args()
task = ARGS.task

with tf.variable_scope(tf.get_variable_scope()):
    input = tf.placeholder(tf.float32, shape=[None, None, None, 3])
    target = tf.placeholder(tf.float32, shape=[None, None, None, 3])
    reflection = tf.placeholder(tf.float32, shape=[None, None, None, 3])
    issyn = tf.placeholder(tf.bool, shape=[])

    # build the model
    network = build(input, True, 64)
    transmission_layer, reflection_layer = tf.split(network, num_or_size_splits=2, axis=3)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
ckpt = tf.train.get_checkpoint_state(task)
print("[i] contain checkpoint: ", ckpt)

saver_restore = tf.train.Saver([var for var in tf.trainable_variables() if 'discriminator' not in var.name])
print('loaded ' + ckpt.model_checkpoint_path)
saver_restore.restore(sess, ckpt.model_checkpoint_path)
# Please replace with your own test image path
test_path = ["./test_images/"]
val_names = prepare_data_test(test_path)

if not os.path.isdir('./test_results'):
    os.makedirs('./test_results')

for val_path in val_names:
    testind = val_path.split('/')[-1].split('.')[-2]
    if not os.path.isfile(val_path):
        continue
    img = cv2.imread(val_path)
    input_image = np.expand_dims(np.float32(img), axis=0) / 255.0
    st = time.time()
    output_image_t, output_image_r = sess.run([transmission_layer, reflection_layer], feed_dict={input: input_image})
    print("Test time %.3f for image %s" % (time.time() - st, val_path))
    output_image_t = np.minimum(np.maximum(output_image_t, 0.0), 1.0) * 255.0
    output_image_r = np.minimum(np.maximum(output_image_r, 0.0), 1.0) * 255.0

    cv2.imwrite("./test_results/%s_input.png" % (testind), img)
    cv2.imwrite("./test_results/%s_t_output.png" % (testind), np.uint8(output_image_t[0, :, :, 0:3]))  # output transmission layer
    cv2.imwrite("./test_results/%s_r_output.png" % (testind), np.uint8(output_image_r[0, :, :, 0:3]))  # output reflection layer

    # mse = ((gt - t) ** 2).mean()
    # psnr = 10 * log10(1 / mse)