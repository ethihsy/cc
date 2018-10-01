import os
from util import *
from estimator import *
import numpy as np
import random
import tensorflow as tf
import time
slim = tf.contrib.slim
tf.reset_default_graph()

batch, itr, nn = 10, 50, 1
dim, hd1, hd2 = 784, 400, 200

k   = 10
tau = tf.placeholder(tf.float32)
lr  = tf.placeholder(tf.float32)
x   = tf.placeholder(tf.float32,[batch, dim])

def nets(estimator, discrete):
    h  = slim.fully_connected(x, hd1)
    _z = slim.fully_connected(h, hd2, activation_fn=None)
    z  = tf.reshape(_z, [-1,k])
    u  = tf.random_uniform(tf.shape(z))
    
    _s = z - tf.log(-tf.log(u))
    s  = tf.one_hot(tf.argmax(_s, -1), k) if discrete else tf.nn.softmax(_s/tau)
    s  = tf.reshape(s, [-1, hd2])

    y   = slim.stack(s, slim.fully_connected, [hd1])
    y   = slim.stack(y, slim.fully_connected, [dim], activation_fn=None)
    nll = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.tile(x,[nn,1]), logits=y)
    nll = tf.reduce_sum(nll, 1, keepdims=True)
    
    zz   = tf.reshape(tf.nn.softmax(tf.reshape(_z,[-1,k])), [bb, hd2])
    kld  = tf.reduce_mean(tf.reduce_sum(zz * (tf.log(tf.cast(k,tf.float32)*zz+eps)), 1))  
    elbo = tf.reduce_mean(nll) + kld
    l    = estimator(nll, z, s, tau, slim.get_variables(), 
                     x, nn, k, u, hd2, h) if discrete else 0.
    return elbo, l


def fit_model(filename, steps,  _lr, cate, dataset, sta):
    data_train, data_val, data_test = bomniglot(sta) if dataset=='OMNIGLOT' else bmnist(sta)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())        
        saver = tf.train.Saver()
        directory = 'VAE/'+dataset+'/'+str(cate)+'/'+filename
        rec_ls = np.empty([3, steps*10])

        t = 1.
        for i in xrange(steps*1000):
            batch_ = np.reshape(random.sample(data_train, batch), [batch,-1])
            if not sta:
                batch_ = (np.random.uniform(0.,1.,[batch,dim])<batch_).astype(float)

            dic = {x:batch_, tau:t, lr:_lr}
            res = sess.run([loss, train, aa], dic)
            if i%100==1:
                rec_ls[0, i/100] = res[0]
                rec_ls[2, i/100] = res[-1]
                batch_ = np.reshape(random.sample(data_test, batch), [batch,-1])
                dic_ = {x: batch_, tau:1e-3}
                rec_ls[1, i/100]  = sess.run(loss,  dic_)

            if i%1000==1:
                t = np.maximum(np.exp(-i*1e-5), 0.5)

            if i==steps*100 or i==steps*500 or i==steps*900:
                save_path = saver.save(sess, directory+"/"+str(i)+"_model.ckpt")

        if not os.path.exists(directory):
            os.makedirs(directory)
        np.save(directory+'/loss_rec', rec_ls)
        save_path = saver.save(sess, directory+"/model.ckpt")

discrete = True
loss, esti = nets(perc_rm, discrete)
opt  = tf.train.AdamOptimizer(learning_rate=lr)
grad = update_grad(opt, slim.get_variables(), discrete, loss, esti)
train= opt.apply_gradients(grad)
aa = esti[-1]


for learning_rate in [3e-4]:
    for cate in [k]:
        fit_model('pc_rm3_.001_1_'+str(learning_rate),
                  itr, learning_rate, cate, 'MNIST', True)
        for dataset in ['OMNIGLOT','dy_MNIST']:
            fit_model('pc_rm3_.001_1_'+str(learning_rate),
                      itr, learning_rate, cate, dataset, False)

