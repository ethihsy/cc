
# coding: utf-8

# In[ ]:

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
Bernoulli = tf.contrib.distributions.Bernoulli

data = input_data.read_data_sets("MNIST_data/", one_hot=True)
for i in range(5000):
    data.validation.images[i] = (np.random.uniform(0.,1.,784)<data.validation.images[i]).astype(float)
for i in range(10000):
    data.test.images[i] = (np.random.uniform(0.,1.,784)<data.test.images[i]).astype(float)


# In[ ]:

def mul_b(p, w):
    p = tf.concat([p, tf.ones([tf.shape(p)[0],1])], 1)
    return tf.nn.sigmoid(tf.matmul(p, w))

def bandwidth(p):
    u = tf.tile(tf.reduce_mean(tf.reshape(p,[ns,-1,nn]),0),[ns,1])
    s = tf.sqrt(tf.reduce_sum(tf.reshape(tf.square(p-u),[ns,-1,nn]),0)/(tf.cast(ns,tf.float32)-1.))
    bw = 1.06 * s / tf.pow(tf.cast(ns,tf.float32), 0.2)
    return tf.stop_gradient(tf.tile(bw, [ns,1]))

def dirac(p, bw):
    return tf.check_numerics(1./(bw * np.sqrt(2*np.pi) + eps) * tf.exp(.5 * tf.square(p/(bw+eps))), 'dirac')

def straight_through(p, z):
    return tf.stop_gradient(tf.ceil(p-z) - p) + p

def step_sample(p, z, dbw):
    return tf.stop_gradient(tf.ceil(p-z) - dbw*p) + tf.stop_gradient(dbw)*p
    
def smooth_sample(p, z, bw):
    s = (p-z) / (tf.sqrt(2*tf.square(bw)) + eps)
    return tf.check_numerics(0.5 * (1 + tf.erf(s)), 'erf')


# In[ ]:

lr  = tf.placeholder(tf.float32)
eps = 1e-7
dim = 392
nn  = 200
ns  = tf.placeholder(tf.int32)
ST = tf.placeholder(tf.bool)


wxh  = tf.Variable(tf.random_normal([dim+1,nn], stddev=np.sqrt(2.55e-3)))
whh  = tf.Variable(tf.random_normal([nn+1, nn], stddev=np.sqrt(5e-3)))
why  = tf.Variable(tf.random_normal([nn+1,dim], stddev=np.sqrt(5e-3)))


y_  = tf.placeholder(tf.float32, [None, dim])
x   = tf.placeholder(tf.float32, [None, dim])

h  = mul_b(x,  wxh)
nh = tf.tile(h, [ns,1])
sd = tf.random_uniform(tf.shape(nh))
bw = bandwidth(nh-sd)
dw = dirac(nh-sd,bw)
#sh = step_sample(nh, sd, dw)
sh = tf.cond(ST, lambda: straight_through(nh, sd),
                 lambda: smooth_sample(nh, sd, bw))

#h2 = mul_b(sh, whh)
#sd2 = tf.random_uniform(tf.shape(h2))
#bw2 = bandwidth(h2-sd2)
#sh2 = tf.cond(ST, lambda: straight_through(h2, sd2),
#                  lambda: smooth_sample(h2, sd2, bw2))

y  = mul_b(sh,why)
yy  = tf.pow(y+eps, tf.tile(y_,[ns,1])) * tf.pow(1-y+eps, 1-tf.tile(y_,[ns,1]))
ye  = tf.reduce_mean(tf.reshape(yy,[ns,-1,dim]),0)
nll = tf.reduce_mean(tf.reduce_sum(-tf.log(ye),1))

opt = tf.train.AdamOptimizer(lr)
train = opt.minimize(nll)


# In[ ]:

tsh = Bernoulli(probs=nh,dtype=tf.float32).sample()
#th2 = mul_b(tsh, whh)
#tsh2 = Bernoulli(probs=th2,dtype=tf.float32).sample()
ty  = mul_b(tsh,why)
tyy  = tf.pow(ty+eps, tf.tile(y_,[ns,1])) * tf.pow(1-ty+eps, 1-tf.tile(y_,[ns,1]))
tye  = tf.reduce_mean(tf.reshape(tyy,[ns,-1,dim]),0)
tnll = tf.reduce_mean(tf.reduce_sum(-tf.log(tye),1))


# In[ ]:

def fit_model(steps, filename, _lr, _ns, _ST):
    train_loss = np.empty((1000, steps/1000))
    test_loss = []
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(steps):
            batch_   = data.train.next_batch(24, shuffle=True)[0]
            batch_xs = (np.random.uniform(0.,1.,[24,392])<batch_[:, 0:392]).astype(float)
            batch_ys = (np.random.uniform(0.,1.,[24,392])<batch_[:, 392:784]).astype(float)
            res = sess.run([nll, train], {x: batch_xs, y_: batch_ys, ns: _ns, lr: _lr, ST:_ST}) 

            train_loss[i%1000, i/1000] = res[0]
            if (i+1)%1000==0:
                batch_   = data.test.next_batch(100, shuffle=True)[0]
                batch_xs = batch_[:, 0:392]
                batch_ys = batch_[:, 392:784]
                res = sess.run(tnll, {x: batch_xs, y_: batch_ys, ns: 100, lr: _lr, ST:_ST})  
                test_loss.append(res)

        np.save(filename, [train_loss, test_loss])
    return train_loss


# In[ ]:

erf10 = fit_model(300000, "st10_2", 1e-3, 10, True)

