#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorlayer_mock as tl
from base_layer import Input, Dropout, Dense
import numpy as np

def disciminator(inputs_shape):
    innet = Input(inputs_shape)
    net = Dense(n_units=32, act=tf.nn.relu)(innet)
    net = Dropout(keep=0.8, seed=1)(net)
    net1 = Dense(n_units=1, act=tf.nn.relu)(net)
    net2 = Dense(n_units=5, act=tf.nn.relu)(net)
    D = tl.Model(inputs=innet, outputs=[net1, net2])
    return D

inputs = tf.placeholder(shape=[None, 100], dtype=tf.float32)
D = disciminator(inputs_shape=[None, 100])
outputs_train = D(inputs, is_train=True)
outputs_test = D(inputs, is_train=False)
# D2 = tl.Model(reuse=True, is_train=False, model=D)

# D.print_weights(False)
# D.count_weights()
# D.weights

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# D.print_layers()
#D.print_weights(True)

real_inputs = np.ones((5, 100))
real_outputs_train = sess.run(outputs_train, feed_dict={inputs: real_inputs})
real_outputs_test = sess.run(outputs_test, feed_dict={inputs: real_inputs})
print(real_outputs_train)
print(real_outputs_test)

# outputs = sess.run(D2.outputs, feed_dict={inputs: images})
