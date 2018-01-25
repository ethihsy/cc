import tensorflow as tf
eps = 1e-7


def straight_through(p, z, t, c=0,d=0,k=0):
    s = tf.ceil(p-z)
    return tf.stop_gradient(s - p) + p

def anneal_straight_through(p, z, t, c=0,d=0,k=0):
    s = tf.ceil(p-z)
    pp = tf.clip_by_value(1./t*(p-z)+0.5, 0., 1.)
    return tf.stop_gradient(s-pp)+pp

def gumbel(p, z, t, c=0,d=0,k=0):
    s = tf.log(p/(1-p+eps)+eps)-tf.log(z/(1-z+eps)+eps)
    return tf.nn.sigmoid(s/t)

def st_gumbel(p, z, t, c=0,d=0,k=0):
    s = tf.log(p/(1-p+eps)+eps)-tf.log(z/(1-z+eps)+eps)
    s = tf.nn.sigmoid(s/t)
    s_ = tf.ceil(p-z)
    return tf.stop_gradient(s_ - s) + s

def gumbel_cat(p,z,t, c=0,d=0,k=0):
    g = -tf.log(-tf.log(z+eps)+eps)
    s = tf.log(p/(1-p+eps)+eps)+g
    return tf.nn.softmax(s/t)

def discrete(p,z,t, c=0,d=0,k=0):
    return tf.ceil(p-z)

def transform(p, u, t, c=0,d=5.,k=0):
    u = tf.clip_by_value(u,1-p,1.)
    s = 1./d*tf.log((u+p-1)/(p+eps)*(tf.exp(d)-1.)+1.)
    return s*tf.cast(u>=1-p, tf.float32)

def transform0(p, u, t, c=0, d=5.,k=0):
    uu = tf.clip_by_value(u,0,1-p)
    s = 1 - 1./d*tf.log(-(uu*(tf.exp(d)-1))/(1-p) + tf.exp(d))
    return s*tf.cast(u<=1-p, tf.float32) + tf.cast(u>1-p,tf.float32)

def mse(p, x, t=0.1, c=0., d=1, kk=0):
    k = kk*p+.3

    u0b, u1b = 1-p+k, 0.
    w0 = 1 - (1-p)*(tf.exp(d*u1b)-tf.exp(d*u0b)) / (tf.exp(d*p)-tf.exp(d*u0b)+eps)
    w0 = tf.clip_by_value(w0,eps,1-eps)
    w0 = tf.check_numerics(w0,'w0')
    
    x0 = tf.clip_by_value(x,w0,1-eps)
    s0 = 1./d*tf.log((1-x0)/(1-w0)*(tf.exp(d*u1b)-tf.exp(d*u0b))+tf.exp(d*u0b))
    s0 = tf.check_numerics(s0,'s0')    
    s0 = s0*tf.cast(x>w0,tf.float32) + u1b*tf.cast(x<=w0,tf.float32)
    
    u0a, u1a = 1., p-k
    w1 = p*(tf.exp(d*u0a)-tf.exp(d*u1a)) / (tf.exp(d*u0a)-tf.exp(d*(u0a+u1a-p))+eps)
    w1 = tf.clip_by_value(w1,eps,1-eps)
    w1 = tf.check_numerics(w1,'w1')

    x1 = tf.clip_by_value(x,eps,w1)
    s1 = -1./d*tf.log(x1/w1*(tf.exp(d*u1a)-tf.exp(d*u0a))+tf.exp(d*u0a)) + u0a + u1a
    s1 = tf.check_numerics(s1,'s1')
    s1 = s1*tf.cast(x<w1,tf.float32) + u0a*tf.cast(x>=w1,tf.float32)

    s  = s1*tf.cast(x<p+c,tf.float32) + s0*tf.cast(x>p+c,tf.float32)

    return tf.nn.sigmoid((p-s)/t)
