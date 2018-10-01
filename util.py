import tensorflow as tf
import numpy as np
import random
from tensorflow.examples.tutorials.mnist import input_data
import scipy.io
import os


def bmnist(sta):
    data = input_data.read_data_sets("dataset/MNIST/", one_hot=True)
    data_train = data.train.images[5000:]
    data_val = np.concatenate([data.validation.images, data.train.images[:5000]],0)
    data_test= data.test.images
    
    if sta:
        data_train = (np.random.uniform(0.,1.,[50000,784])<data_train).astype(float)
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


def bomniglot(sta):
    omni_raw = scipy.io.loadmat(os.path.join('dataset/OMNIGLOT/omniglot.mat'))
    data_train = np.transpose(omni_raw['data'])
    data_val = data_train[:8070]
    data_train = data_train[8070:]
    data_test  = np.transpose(omni_raw['testdata'])

    data_train  = (np.random.uniform(0.,1.,[16275,784])<data_train).astype(float)
    data_val  = (np.random.uniform(0.,1.,[8070,784])<data_val).astype(float)
    data_test  = (np.random.uniform(0.,1.,[8070,784])<data_test).astype(float)

    return data_train, data_val, data_test

