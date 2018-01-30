import os
from util import *
from estimator import *
import numpy as np
import random
import tensorflow as tf
slim = tf.contrib.slim
tf.reset_default_graph()


tau = tf.placeholder(tf.float32)
lr  = tf.placeholder(tf.float32)
x  = tf.placeholder(tf.float32,[100,784])
cc = tf.placeholder(tf.float32,[200])
kk = tf.placeholder(tf.float32)
dd = tf.placeholder(tf.float32)

def nets(estimator):
    h = slim.stack(x,slim.fully_connected,[400])
    h = slim.fully_connected(h,200,activation_fn=None)
    z = tf.nn.sigmoid(h)
    
    s = tf.reshape(estimator(z,tf.random_uniform(tf.shape(z)),tau, cc,dd,kk), [-1,200])
    h2= slim.stack(slim.flatten(s),slim.fully_connected,[400])    
    y = slim.fully_connected(h2,784,activation_fn=None)
    
    kld = tf.reduce_sum(z*(tf.log(z+eps)-tf.log(.5)) + (1-z)*(tf.log(1-z+eps)-tf.log(.5)), 1)
    nll = tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=y)
    elbo= tf.reduce_sum(nll,1) + kld
    
    return tf.reduce_mean(elbo), elbo, tf.nn.sigmoid(y)



def fit_model(filename, _lr, t, dataset, para=None):
    
    if dataset =='M':
        data_train, data_val, data_test = bmnist()
        dsfd = 'MNIST'
    else:
        data_train, data_val, data_test = bomniglot()
        dsfd = 'OMNIGLOT'

    steps = 100
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())        
        saver = tf.train.Saver()
        train_loss = np.empty((steps))
        val_loss   = np.empty((steps))
        test_loss  = np.empty((steps))
        var_g      = np.empty((steps))

        batch = 100
        d = 1.

        c_ = np.asarray([1.]*100+[0.]*100)
        c = c_
        k = 0.

        for i in xrange(steps*1000):
            batch_ = np.reshape(random.sample(data_train, batch), [batch,-1])
            batch_ = (np.random.uniform(0.,1.,[batch,784])<batch_).astype(float)
            _, res = sess.run([train,loss],{x:batch_, tau:t, lr:_lr, cc:c, kk:k, dd:d})
            

            if i%1000==1:
                ind = i/1000 
                train_loss[ind] = res
                var_g[ind] = sess.run(vg,{x:batch_, tau:t, lr:_lr, cc:c, kk:k, dd:d})
                batch_ = np.reshape(random.sample(data_val, batch), [batch,-1])
                val_loss[ind]  = sess.run(loss,{x:batch_, tau:0.001, lr:_lr, cc:np.zeros([200]), kk:1.,dd:3.})
                batch_ = np.reshape(random.sample(data_test, batch), [batch,-1])
                test_loss[ind] = sess.run(loss,{x:batch_, tau:0.001, lr:_lr, cc:np.zeros([200]), kk:1.,dd:3.})


                if filename[:2]=='MX':
                    d = 3. - 2.*np.exp(-1e-5*para*i)
                    k = 1. - np.exp(-6e-5*i)
                    c = np.exp(-3e-5*i)*c_

                else:
                    t = np.maximum(np.exp(-3e-5*i),0.5)


        directory = 'VAE/'+dsfd+'/'+filename
        if not os.path.exists(directory):
            os.makedirs(directory)
        np.save(directory+'/loss_rec', [train_loss, val_loss, test_loss, var_g])
        save_path = saver.save(sess, directory+"/model.ckpt")




loss, vl, _ = nets(mse)
vg = [jacobian(vl, slim.get_model_variables()[z]) for z in range(4)]
vg = [tf.reduce_sum(tf.square(z-tf.reduce_mean(z,0))) for z in vg]
vg = tf.reduce_sum(vg) / 100. / (784*400+400+400*200+200) 
train=tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, var_list=slim.get_model_variables())



for q1 in ['M','O']:
#    for q2 in ['200_0_0','100_100_0','0_200_0',
#               '100_0_100','0_0_200',
#               '50_75_75','100_50_50','150_25_25',
#               '75_50_75','50_100_50','25_150_25']:
#        q3 = str.split(q2,'_')
#        fit_model('MX1e-3_'+q2, 1e-3, .1, q1, np.asarray([1]*int(q3[0])+[0]*int(q3[1])+[-1]*int(q3[2])))

 #       for q3 in [3.,7.]:
#            fit_model('MX1e-3_'+str(q3), 1e-3, .1, q1, q3)
    fit_model('GB1e-3', 1e-3, 1., q1)


