import tensorflow as tf
slim = tf.contrib.slim
import numpy as np
from util import *

binCon = tf.contrib.distributions.RelaxedBernoulli
log_pdf = lambda z, q:  q*tf.log(z+eps) 
logit = lambda a: tf.log(a/(1-a+eps)+eps)
eps = 1e-7
bb = 10 # batch size

def forward_pass(y, ww, x, ns, stop=True):
    for i, w in enumerate(ww):
        if stop:
            w = tf.stop_gradient(w)
        if i%2==0:
            y = tf.matmul(y, w)
        elif i==len(ww)-1:
            y  = y + w
            l_ = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.tile(x,[ns,1]), logits=y)
        else:
            y = tf.nn.relu(y+w)
    return tf.reduce_sum(l_, 1, keepdims=True)

def update_grad(opt, w, discrete, *arg):
    grad = []
    gt = opt.compute_gradients(arg[0], w)
    for i in gt:
        grad.append([i[0], i[1]])

    if discrete:
        gd = opt.compute_gradients(arg[1][0], w[:4])
        for i in range(4):
            grad[i][0] = gd[i][0]
    return grad

#-----------------------------------------------------------

def sf(l, p, s, step, w, x, ns, k, u, nn):
    p = tf.reshape(tf.nn.softmax(p), [-1, nn])
    f = tf.reduce_sum(tf.stop_gradient(l) * log_pdf(p, s), 1)
    return tf.reduce_mean(f), tf.constant(0.)

def muprop(l, p, s, t, w, x, ns, k, *arg):
    p  = tf.reshape(tf.nn.softmax(p), [bb, -1])
    l_ = forward_pass(p, w, x, ns)
    f_ = tf.gradients(tf.reduce_mean(l_), p)[0]
    ll = l_ + tf.reduce_sum(f_*(s-p), 1, keepdims=True)

    f = tf.reduce_sum(tf.stop_gradient(l-ll) * log_pdf(p,s), 1)
    m = tf.reduce_sum(tf.stop_gradient(f_)*p, 1)
    return tf.reduce_mean(f+m), tf.constant(0.)

def rebar(l, p, s, t, w, x, ns, k, u, nn):
    _s = gumbel(p, t, k, nn, None, False)
    s_ = gumbel(p, t, k, nn,    s, False, u)
    ss = tf.concat([_s,s_],0)
    ll = forward_pass(ss, w, x, 2*ns)
    _l, l_ = ll[:bb*ns], ll[bb*ns:]
    
    p = tf.reshape(tf.nn.softmax(p), [bb, -1])
    f = tf.reduce_sum(tf.stop_gradient(l-l_)  * log_pdf(p,s), 1, keepdims=True)  - l_

    return tf.reduce_mean(f+_l), tf.constant(0.)

def gumbel(p, t, cat, h_dim, s=None, discrete=False, su=None):
    u  = tf.random_uniform(tf.shape(p))
    zz = tf.log(u+eps)
    if s is None:
        s = p - tf.log(-zz+eps)
    else:
        su = tf.log(su+eps)
        p  = tf.reshape(tf.nn.softmax(p), [-1,cat])
        s  = tf.where(tf.reshape(s,[-1,cat])>.5, -tf.log(-su+eps), -tf.log(-zz/(p+eps) - su + eps))
    s = tf.check_numerics(s,'sss')
    s = tf.reshape(tf.nn.softmax(s/t), [-1, h_dim])
    return tf.ceil(s-.5) if discrete else s

def perc(l, p, _s, step, w, x, ns, k, u, nn):
    def lk(i, n, dd, s, x, w):
        ss = tf.concat([s[:,:i], tf.ones([bb,1]), tf.zeros([bb,k-1]), s[:,i+k:]], 1)
        for ii in range(1,k):
            s0 = tf.concat([s[:,:i], tf.zeros([bb,ii]), tf.ones([bb,1]), tf.zeros([bb,k-ii-1]), s[:,i+k:]], 1)
            ss = tf.concat([ss, s0], 0)
        nl = forward_pass(ss, w, x, k)
        ll = tf.transpose(tf.reshape(nl, [-1,bb]))
        dd = dd - tf.pad(ll*step, [[0,0],[i,nn-i-k]])
        return i+k, n, dd, s, x, w

    ns_,i = tf.constant(nn), tf.constant(0)
    uu = tf.reshape(p - tf.log(-tf.log(u+eps)+eps), [-1,nn])
    ll = tf.while_loop(lambda i,n,*arg: i<n, lk, [i, ns_, uu, _s, x, w[4:]],
                       parallel_iterations=24)
    
    tt = tf.reshape(tf.one_hot(tf.argmax(tf.reshape(ll[2], [-1,k]),-1), k), [-1,nn])
    pt = tf.stop_gradient(_s-tt)

    return tf.reduce_mean(tf.reduce_sum(pt*tf.reshape(p,[bb,nn]), 1))/step, tf.reduce_mean(tf.reduce_sum(tf.abs(pt),1))

def perc_rm(l, p, _s, step, w, x, ns, k, u, nn, hh):

    # loss augmented inference using the local expectation
    def lk(i, n, dd, s, x, w):
        ss = tf.concat([s[:,:i], tf.ones([bb,1]), tf.zeros([bb,k-1]), s[:,i+k:]], 1)
        for ii in range(1,k):
            s0 = tf.concat([s[:,:i], tf.zeros([bb,ii]), tf.ones([bb,1]), tf.zeros([bb,k-ii-1]), s[:,i+k:]], 1)
            ss = tf.concat([ss, s0], 0)
        nl = forward_pass(ss, w, x, k)
        ll = tf.transpose(tf.reshape(nl, [-1,bb]))
        dd = dd - tf.pad(ll*step, [[0,0],[i,nn-i-k]])
        return i+k, n, dd, s, x, w

    ns_,i = tf.constant(nn), tf.constant(0)
    uu = tf.reshape(p - tf.log(-tf.log(u+eps)+eps), [-1,nn])
    ll = tf.while_loop(lambda i,n,*arg: i<n, lk, [i, ns_, uu, _s, x, w[4:]],
                       parallel_iterations=24)
    
    # update categories
    tt = tf.reshape(tf.one_hot(tf.argmax(tf.reshape(ll[2], [-1,k]),-1), k), [-1,nn])
    pt = tf.stop_gradient(_s-tt)

    # margin constraint 1
    lt = tf.reduce_sum(tf.reshape((-ll[2]+uu)*tt, [-1,k]),-1)
    lw = tf.reshape(tf.tile(l,[1,nn/k]),[-1])
    xi = lw-lt + (tf.reduce_sum(tf.reshape(_s,[-1,k])*p,1) - tf.reduce_sum(tf.reshape(tt,[-1,k])*p,1))
    xi = tf.expand_dims(tf.cast(xi>0, tf.float32), -1)
    pt = tf.reshape(tf.reshape(pt,[-1,k])*xi, [bb,nn])

    # margin constraint 2 (relative maximal margin)
    aw = tf.one_hot(tf.argmax(tf.reshape(-ll[2]+2*uu,[-1,k]),-1),k)
    la = tf.reduce_sum(aw*tf.reshape(-ll[2]+uu,[-1,k]),-1)
    tp = tf.reduce_sum(tf.reshape(tt,[-1,k])*p,-1,keepdims=True) 
    rm = (tp - tf.reduce_sum(aw*p,-1,keepdims=True))
    xia= rm-tf.expand_dims(la-lt,-1)
    rr = tf.where(xia>0, tf.ones([bb*nn/k,1]), tf.zeros([bb*nn/k,1])) * rm

    # step size (\lambda and \Omega)
    dish = tf.reduce_sum(hh**2,1,keepdims=True)
    dis = tf.reshape(xi,[bb,-1]) / (dish+1)
    dis = tf.minimum(dis,0.001)
    dis = tf.stop_gradient(dis) / (3e-4)
    
    dis2= tf.reshape(xia, [bb,-1]) / (dish+1) 
    dis2= tf.minimum(dis2, 1)
    dis2= tf.stop_gradient(dis2) / (3e-4)
    rr  = tf.reduce_mean(tf.reduce_sum(tf.reshape(rr,[bb,-1])*dis2,1))

    return tf.reduce_mean(tf.reduce_sum(pt*tf.reshape(p,[bb,nn]), 1))*dis+rr, tf.reduce_mean(tf.reduce_sum(tf.abs(pt),1))


