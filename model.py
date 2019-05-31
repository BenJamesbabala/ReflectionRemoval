import tensorflow.contrib.slim as slim
import tensorflow as tf
from vgg19 import build_vgg19
import numpy as np

###################Reflection Removal Model#######################
def lrelu(x):
    return tf.maximum(x*0.2,x)

def relu(x):
    return tf.maximum(0.0,x)

def identity_initializer():
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        array = np.zeros(shape, dtype=float)
        cx, cy = shape[0]//2, shape[1]//2
        for i in range(np.minimum(shape[2],shape[3])):
            array[cx, cy, i, i] = 1
        return tf.constant(array, dtype=dtype)
    return _initializer

def nm(x):
    w0=tf.Variable(1.0,name='w0')
    w1=tf.Variable(0.0,name='w1')
    return w0*x+w1*slim.batch_norm(x)

def build(input, hyper, channel):
    if hyper:
        print("[i] Hypercolumn ON, building hypercolumn features ... ")
        vgg19_features=build_vgg19(input[:,:,:,0:3]*255.0)
        for layer_id in range(1,6):
            vgg19_f = vgg19_features['conv%d_2'%layer_id]
            input = tf.concat([tf.image.resize_bilinear(vgg19_f,(tf.shape(input)[1],tf.shape(input)[2]))/255.0,input], axis=3)
    else:
        vgg19_features=build_vgg19(input[:,:,:,0:3]*255.0)
        for layer_id in range(1,6):
            vgg19_f = vgg19_features['conv%d_2'%layer_id]
            input = tf.concat([tf.image.resize_bilinear(tf.zeros_like(vgg19_f),(tf.shape(input)[1],tf.shape(input)[2]))/255.0,input], axis=3)
    net=slim.conv2d(input,channel,[1,1],rate=1,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv0')
    net=slim.conv2d(net,channel,[3,3],rate=1,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv1')
    net=slim.conv2d(net,channel,[3,3],rate=2,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv2')
    net=slim.conv2d(net,channel,[3,3],rate=4,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv3')
    net=slim.conv2d(net,channel,[3,3],rate=8,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv4')
    net=slim.conv2d(net,channel,[3,3],rate=16,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv5')
    net=slim.conv2d(net,channel,[3,3],rate=32,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv6')
    net=slim.conv2d(net,channel,[3,3],rate=64,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv7')
    net=slim.conv2d(net,channel,[3,3],rate=1,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv9')
    net=slim.conv2d(net,3*2,[1,1],rate=1,activation_fn=None,scope='g_conv_last') # output 6 channels --> 3 for transmission layer and 3 for reflection layer
    return net

##################Loss##################

def compute_l1_loss(input, output):
    return tf.reduce_mean(tf.abs(input - output))


def compute_percep_loss(input, output, reuse=False):
    vgg_real = build_vgg19(output * 255.0, reuse=reuse)
    vgg_fake = build_vgg19(input * 255.0, reuse=True)
    p0 = compute_l1_loss(vgg_real['input'], vgg_fake['input'])
    p1 = compute_l1_loss(vgg_real['conv1_2'], vgg_fake['conv1_2']) / 2.6
    p2 = compute_l1_loss(vgg_real['conv2_2'], vgg_fake['conv2_2']) / 4.8
    p3 = compute_l1_loss(vgg_real['conv3_2'], vgg_fake['conv3_2']) / 3.7
    p4 = compute_l1_loss(vgg_real['conv4_2'], vgg_fake['conv4_2']) / 5.6
    p5 = compute_l1_loss(vgg_real['conv5_2'], vgg_fake['conv5_2']) * 10 / 1.5
    return p0 + p1 + p2 + p3 + p4 + p5


def compute_exclusion_loss(img1, img2, level=1):
    gradx_loss = []
    grady_loss = []

    for l in range(level):
        gradx1, grady1 = compute_gradient(img1)
        gradx2, grady2 = compute_gradient(img2)
        alphax = 2.0 * tf.reduce_mean(tf.abs(gradx1)) / tf.reduce_mean(tf.abs(gradx2))
        alphay = 2.0 * tf.reduce_mean(tf.abs(grady1)) / tf.reduce_mean(tf.abs(grady2))

        gradx1_s = (tf.nn.sigmoid(gradx1) * 2) - 1
        grady1_s = (tf.nn.sigmoid(grady1) * 2) - 1
        gradx2_s = (tf.nn.sigmoid(gradx2 * alphax) * 2) - 1
        grady2_s = (tf.nn.sigmoid(grady2 * alphay) * 2) - 1

        gradx_loss.append(
            tf.reduce_mean(tf.multiply(tf.square(gradx1_s), tf.square(gradx2_s)), reduction_indices=[1, 2, 3]) ** 0.25)
        grady_loss.append(
            tf.reduce_mean(tf.multiply(tf.square(grady1_s), tf.square(grady2_s)), reduction_indices=[1, 2, 3]) ** 0.25)

        img1 = tf.nn.avg_pool(img1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        img2 = tf.nn.avg_pool(img2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    return gradx_loss, grady_loss


def compute_gradient(img):
    gradx = img[:, 1:, :, :] - img[:, :-1, :, :]
    grady = img[:, :, 1:, :] - img[:, :, :-1, :]
    return gradx, grady