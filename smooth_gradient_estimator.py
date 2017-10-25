import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
Bernoulli = tf.contrib.distributions.Bernoulli

data = input_data.read_data_sets("MNIST_data/", one_hot=True)
for i in range(5000):
    data.validation.images[i] = (np.random.uniform(0.,1.,784)<data.validation.images[i]).astype(float)
for i in range(10000):
    data.test.images[i] = (np.random.uniform(0.,1.,784)<data.test.images[i]).astype(float)


#------------------------------------------------------------------------
eps = 1e-7
dim = 392         # observed nodes
nn  = 200         # hidden nodes
batch = 24
ns  = 10          # samples

lr  = tf.placeholder(tf.float32)
opt = tf.train.AdamOptimizer(lr)
optbw = tf.train.AdamOptimizer(lr)
coef_bw = .01     # lambda
const_bw = 10     # constraint C


#------------------------------------------------------------------------
def mul_b(p, w, af=None):
    p = tf.concat([p, tf.ones([tf.shape(p)[0],1])], 1)
    return tf.matmul(p,w) if af is None else af(tf.matmul(p, w))

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

def opt_bw(l, tl, g):
    bwg = jacobian(tf.reshape(l,[ns,-1,1]), wx)
    bwl = tf.reshape(tf.square(tf.reshape(bwg,[-1,nn]) - tf.tile(g[0][0],[ns,1])), [ns,dim+1,nn])
    bwl = tf.reduce_sum(bwl, 0) / tf.cast(ns,tf.float32) / (tf.cast(ns,tf.float32)-1)
    bias = coef_bw * tf.square(tf.reduce_mean(tf.square(l-tl)-const_bw))
    bwgd = optbw.compute_gradients(bwl + bias, bw)
    return optbw.apply_gradients(bwgd)


#------------------------------------------------------------------------
wx = tf.Variable(tf.random_normal([dim+1,nn], stddev=np.sqrt(3.37e-3)))
wy = tf.Variable(tf.random_normal([nn+1,dim], stddev=np.sqrt(3.37e-3)))
bw = tf.Variable(tf.ones([nn])*.05)
y_ = tf.placeholder(tf.float32, [batch, dim])
x  = tf.placeholder(tf.float32, [batch, dim])

h  = mul_b(x, wx, tf.nn.sigmoid)
nh = tf.tile(h, [ns,1])
sd = tf.random_uniform(tf.shape(nh))
sh = smooth_sample(nh, sd, bw)

y  = mul_b(sh, wy)
yy = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.tile(y_,[ns,1]),logits=y),1)
nll = tf.reduce_mean(yy)

tsh = tf.ceil(nh-sd)
ty  = mul_b(tsh, wy)
tyy = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.tile(y_,[ns,1]),logits=ty),1)
tnll = tf.reduce_mean(tyy)


#------------------------------------------------------------------------
gd  = opt.compute_gradients(nll, [wx,wy])
train = opt.apply_gradients(gd)
train_bw = opt_bw(yy, tyy, gd)

#------------------------------------------------------------------------
def fit_model(steps, filename, _lr):
    train_loss = np.empty((1000, steps/1000))
    test_loss = []
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(steps):
            batch_   = data.train.next_batch(batch, shuffle=True)[0]
            batch_xs = (np.random.uniform(0.,1.,[batch,dim])<batch_[:, 0:392]).astype(float)
            batch_ys = (np.random.uniform(0.,1.,[batch,dim])<batch_[:, 392:784]).astype(float)
            res = sess.run([nll, train, train_bw], {x: batch_xs, y_: batch_ys, lr: _lr}) 

            train_loss[i%1000, i/1000] = res[0]
            if (i+1)%1000==0:
                batch_   = data.test.next_batch(batch, shuffle=True)[0]
                batch_xs = batch_[:, 0:392]
                batch_ys = batch_[:, 392:784]
                res = sess.run(tnll, {x: batch_xs, y_: batch_ys, lr: _lr})  
                test_loss.append(res)

        np.save(filename, [train_loss, test_loss])
    return train_loss


erf10 = fit_model(300000, "opt_10_p01", 1e-3)
