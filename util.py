import tensorflow as tf
import numpy as np
import random
from tensorflow.examples.tutorials.mnist import input_data
import scipy.io
import os
import matplotlib.pyplot as plt


def bmnist():
    data = input_data.read_data_sets("dataset/MNIST/", one_hot=True)
    data_train = data.train.images[5000:]
    data_val = np.concatenate([data.validation.images, data.train.images[:5000]],0)
    data_test= data.test.images
    data_val = (np.random.uniform(0.,1.,[10000,784])<data_val).astype(float)
    data_test = (np.random.uniform(0.,1.,[10000,784])<data_test).astype(float)
    
    return data_train, data_val, data_test


def jacobian(y_flat, x):
    n = y_flat.shape[0]
    loop_vars = [
        tf.constant(0, tf.int32),
        tf.TensorArray(tf.float32, size=n),
    ]
    _, jacobian = tf.while_loop(
        lambda j, _: j < n,
        lambda j, result: (j+1, result.write(j, tf.gradients(y_flat[j], x))),
        loop_vars)
    return tf.squeeze(jacobian.stack())


def bomniglot():
    omni_raw = scipy.io.loadmat(os.path.join('dataset/OMNIGLOT/omniglot.mat'))
    data_train = np.transpose(omni_raw['data'])
    data_val = data_train[:8070]
    data_train = data_train[8070:]
    data_test  = np.transpose(omni_raw['testdata'])
    data_val  = (np.random.uniform(0.,1.,[8070,784])<data_val).astype(float)
    data_test  = (np.random.uniform(0.,1.,[8070,784])<data_test).astype(float)

    return data_train, data_test, data_test


def draw(batch_xs, ss, ii, half):
    fig, ax = plt.subplots(1,1, figsize=(18,10))
    up = np.concatenate([j for j in [np.reshape(i,[14*half,28]) for i in batch_xs]], 1)

    for si, s in enumerate(ss):
        n, nn = 5, 20
        p1 = np.concatenate([j for j in [np.reshape(i,[14*half,28]) for i in s]], 1)
        p1 = np.concatenate(np.split(np.concatenate([up,p1], 0), 1, 1), 0) if half==1 else p1
        qw = np.ones([nn*28,5])

        for i in range(nn):
            pp1 = p1[:,i*28*n:i*28*n+28*n] if i==0 else np.concatenate([pp1, p1[:,i*28*n:i*28*n+28*n]],0)
    
        a = np.concatenate([pp1,qw],1) if si==0 else np.concatenate([a,pp1,qw],1)

    plt.imshow(a, cmap=plt.cm.gray, interpolation='none')
    plt.savefig(ii)


