#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorlayer as tl
import numpy as np
# import tensorflow.contrib.eager as tfe
# from keras.datasets import cifar10
tf.enable_eager_execution()

def generator(input_shape):
    inputs = tl.layers.Input(input_shape)
    net = tl.layers.Dense(n_units=64, act=tf.nn.elu)(inputs)
    net = tl.layers.Dropout(keep=0.8)(net)
    net2 = tl.layers.Dense(n_units=64, act=tf.nn.elu)(net)

    test = tl.layers.Dense(units=64, activation=tf.nn.elu, name="fc3")
    print(test.weights)
    print(test.input_shape, test.output_shape) # see more from keras: https://tensorflow.google.cn/api_docs/python/tf/keras/layers/Conv2D
    
    # print(net2.output) # AttributeError: 'DeferredTensor' object has no attribute 'output'
    print(net2)
    print(net2.shape, net2.name, net2.dtype)
    print(tl.layers.Dense(n_units=64, act=tf.nn.elu).weights)

    net = tl.layers.Dense(n_units=1)(net2)
    G = tl.Model(inputs=inputs, outputs=[net, net2])
    return G, net2

latent_space_size = 100
G, net2 = generator((None, latent_space_size))
G.print_weights(True)
G.print_layers()
G.count_weights()
print(G.weights)
print(G.outputs) # keras: [<DeferredTensor 'None' shape=(?, ?, 1) dtype=float32>, <DeferredTensor 'None' shape=(?, ?, 64) dtype=float32>]
# print(net2.output) # keras: AttributeError: 'DeferredTensor' object has no attribute 'output'
inputs = np.ones((10, latent_space_size))
outputs = G(inputs, is_train=True)
outputs = G(inputs, is_train=False)

