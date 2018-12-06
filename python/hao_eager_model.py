#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorlayer_mock as tl
from base_layer import Input, Dropout, Dense
import numpy as np
# import tensorflow.contrib.eager as tfe
# from keras.datasets import cifar10

tf.enable_eager_execution()

def generator(inputs, train):
    net = Input()(inputs)
    net = Dense(n_units=64, act=tf.nn.relu)(net, train)
    net = Dropout(keep=0.8, seed=1)(net, train)
    net = Dense(n_units=64, act=tf.nn.relu)(net, train)
    net = Dense(n_units=1, act=tf.nn.relu)(net, train)
    print(net.weights)
    exit()

    G = tl.Model(inputs=inputs, outputs=[net, net2])
    return G, net2

# latent_space_size = 100
# G, net2 = generator((None, latent_space_size))
inputs = np.zeros([100, 100], dtype="float32")
inputs = tf.convert_to_tensor(inputs)
G, net2 = generator(inputs, train=True)
exit()
G.print_weights(True)
G.print_layers()
G.count_weights()
print(G.weights)
print(G.outputs) # keras: [<DeferredTensor 'None' shape=(?, ?, 1) dtype=float32>, <DeferredTensor 'None' shape=(?, ?, 64) dtype=float32>]
# print(net2.output) # keras: AttributeError: 'DeferredTensor' object has no attribute 'output'
inputs = np.ones((10, latent_space_size))
outputs = G(inputs, is_train=True)
outputs = G(inputs, is_train=False)

