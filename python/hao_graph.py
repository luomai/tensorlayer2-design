#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorlayer_mock as tl
from base_layer import Input, Dropout, Dense
import numpy as np

def disciminator(inputs_shape, is_train):
    innet = Input(inputs_shape)
    net = Dense(n_units=32, act=tf.nn.relu)(innet)
    net = Dropout(keep=0.8, seed=1)(net)
    net1 = Dense(n_units=1, act=tf.nn.relu)(net)
    net2 = Dense(n_units=5, act=tf.nn.relu)(net)
    D = tl.Model(inputs=innet, outputs=[net1, net2], is_train=is_train)
    return D

inputs = tf.placeholder(shape=[None, 100], dtype=tf.float32)
D = disciminator(inputs_shape=[None, 100], is_train=True)
outputs = D(inputs)
# D2 = tl.Model(reuse=True, is_train=False, model=D)

# D.print_weights(False)
# D.count_weights()
# D.weights

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# D.print_layers()
#D.print_weights(True)

real_inputs = np.ones((5, 100))
real_outputs = sess.run(outputs, feed_dict={inputs: real_inputs})
print(real_outputs)

# outputs = sess.run(D2.outputs, feed_dict={inputs: images})
