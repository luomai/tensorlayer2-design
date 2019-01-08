#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorlayer_mock as tl
from base_layer import Input, Dropout, Dense
import numpy as np
# import tensorflow.contrib.eager as tfe
# from keras.datasets import cifar10

tf.enable_eager_execution()

def generator(inputs_shape):
    innet = Input(inputs_shape)
    net = Dense(n_units=64, act=tf.nn.relu)(innet)
    net = Dropout(keep=0.8, seed=1)(net)
    net = Dense(n_units=64, act=tf.nn.relu)(net)
    net1 = Dense(n_units=1, act=tf.nn.relu)(net)
    net2 = Dense(n_units=5, act=tf.nn.relu)(net)

    G = tl.Model(inputs=innet, outputs=[net1, net2])
    return G, net2

latent_space_size = 100
G, net2 = generator((None, latent_space_size))
# inputs = np.zeros([100, 100], dtype="float32")
# inputs = tf.convert_to_tensor(inputs)
# G, net2 = generator(inputs, train=True)
# G.print_weights(True)
# G.print_layers()
# G.count_weights()
# print(G.weights)
# print(G.outputs) # keras: [<DeferredTensor 'None' shape=(?, ?, 1) dtype=float32>, <DeferredTensor 'None' shape=(?, ?, 64) dtype=float32>]
# print(net2.output) # keras: AttributeError: 'DeferredTensor' object has no attribute 'output'
inputs = np.ones((10, latent_space_size), dtype="float32")
outputs_train = G(inputs, True)
outputs_test = G(inputs, False)
print(outputs_train)
print(outputs_test)

