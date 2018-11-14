#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorlayer as tl
import numpy as np

def disciminator(inputs, is_train):
    net = tl.layers.Dense(n_units=32, act=tf.nn.elu)(inputs)
    net = tl.layers.Dropout(keep=0.8)(net)
    net1 = tl.layers.Dense(n_units=1)(net)
    net2 = tl.layers.Dense(n_units=5)(net)
    D = tl.Model(inputs=inputs, outputs=[net1, net2], is_train=is_train)
    return D

inputs = tf.placeholder("float32", [None, 100])
D = disciminator(inputs, is_train=True)
D2 = tl.Model(reuse=True, is_train=False, model=D)

D.print_weights(False)
D.count_weights()
D.weights

sess.run(tf.global_variables_initializer())

D.print_layers()
D.print_weights(True)

inputs = np.ones((5, 100))
outputs = sess.run(D.outputs, feed_dict={inputs: images})
outputs = sess.run(D2.outputs, feed_dict={inputs: images})
