import os
from util import *
from estimator import *
import numpy as np
import random
import tensorflow as tf
slim = tf.contrib.slim
tf.reset_default_graph()



eps = 1e-7
ns  = 10
batch = 24
tau = tf.placeholder(tf.float32)
cc = tf.placeholder(tf.float32, [200])
kk = tf.placeholder(tf.float32, [200])
dd = tf.placeholder(tf.float32)

lr = tf.placeholder(tf.float32)
y_ = tf.placeholder(tf.float32, [batch, 392])
x  = tf.placeholder(tf.float32, [batch, 392])



def nets(estimator):
    h = slim.fully_connected(x,200,activation_fn=tf.nn.sigmoid)
    nh= tf.tile(h,[ns,1])
    s = tf.reshape(estimator(nh,tf.random_uniform(tf.shape(nh)),tau,cc,dd,kk), [-1,200])
    h2= slim.fully_connected(slim.flatten(s),200,activation_fn=tf.nn.sigmoid)
    s2= tf.reshape(estimator(h2,tf.random_uniform(tf.shape(h2)),tau,cc,dd,kk), [-1,200])
    y = slim.fully_connected(s2,392,activation_fn=None)
    
    yy  = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.tile(y_,[ns,1]),logits=y),1)
    ye  = -tf.reduce_logsumexp(tf.reshape(-yy,[ns,-1,1]), 0) + tf.log(tf.cast(ns,tf.float32))

    return tf.reduce_mean(ye), ye, tf.reduce_mean(tf.reshape(tf.nn.sigmoid(y),[ns,-1,392]),0)



def fit_model(filename,_lr,t, dataset):
    steps = 50
    if dataset=='M':
        data_train, data_val, data_test = bmnist()
        dsfd = 'MNIST'
    else:
        data_train, data_val, data_test = bomniglot()
        dsfd = 'OMNIGLOT'

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        train_loss = np.empty(steps)
        val_loss   = np.empty(steps) 
        test_loss  = np.empty(steps)
        var_g      = np.empty((steps))        

        d  = 1.
        c_ = np.asarray([1]*75 + [0]*50 + [-1]*75)
        c = c_
        k  = 1. - np.abs(c_) 

        for i in xrange(steps*1000):
            batch_   = np.reshape(random.sample(data_train, batch), [batch,-1])
            batch_xs = (np.random.uniform(0.,1.,[batch,392])<batch_[:, 0:392]).astype(float)
            batch_ys = (np.random.uniform(0.,1.,[batch,392])<batch_[:, 392:784]).astype(float)
            res, _ = sess.run([loss, train], {x: batch_xs, y_: batch_ys, lr: _lr, tau:t, cc:c, kk:k, dd:d})

            if i % 1000 == 1:
                train_loss[i/1000] = res
                var_g[i/1000] = sess.run(vg,{x:batch_xs, y_:batch_ys, tau:t, lr:_lr,cc:c,dd:d,kk:k})   

                batch_ = np.reshape(random.sample(data_val, batch), [batch,-1])
                batch_xs, batch_ys = batch_[:, 0:392], batch_[:, 392:784]
                val_loss[i/1000] = sess.run(loss, {x: batch_xs, y_: batch_ys, tau:1e-3, cc:np.zeros([200]),kk:np.ones([200]),dd:3.})

                batch_ = np.reshape(random.sample(data_test, batch), [batch,-1])
                batch_xs, batch_ys = batch_[:, 0:392], batch_[:, 392:784]
                test_loss[i/1000] = sess.run(loss, {x: batch_xs, y_: batch_ys,tau:1e-3,
                                                    cc:np.zeros([200]),kk:np.ones([200]),dd:3.})

                if filename[:2]=='MX':
                    d = 3. - 2*np.exp(-0.00003*i)
                    k = 1. - np.exp(-0.00005*i)*np.abs(c_)
                    c = np.exp(-0.00003*i)*c_
                else:
                    t = np.maximum(np.exp(-0.00003*i),0.1)

        directory = 'SNN/'+dsfd+'/'+filename
        if not os.path.exists(directory):
            os.makedirs(directory)
        np.save(directory+'/loss_rec', [train_loss, val_loss, test_loss, var_g])
 #       save_path = saver.save(sess, directory+"/model.ckpt")


loss, vl, _ = nets(mse)
vg = [jacobian(vl, slim.get_model_variables()[z]) for z in range(4)]
vg = [tf.reduce_sum(tf.square(z-tf.reduce_mean(z,0))) for z in vg]
vg = tf.reduce_sum(vg)/24./(392*200+200+200*200+200)
train=tf.train.AdamOptimizer(learning_rate=lr).minimize(loss,var_list=slim.get_model_variables())


fit_model('MX1e-3', 1e-3, .1, 'O')
