#!/usr/bin/env python3

import tensorflow as tf

import tensorlayer_mock as tl
from base_layer import InputLayer, MagicalDenseLayer

# print('x :: %s' % (x))
image = tf.placeholder(tf.float32, [None, 28 * 28 * 3])

x = InputLayer()(image)
y = MagicalDenseLayer(tf.constant(10.0), 1000)(x, train=True, reuse=False)
z = MagicalDenseLayer(tf.constant(15.0), 1000)(y, train=True, reuse=False)

print('y :: %s' % (y))
print(y.all_weights)

print('z :: %s' % (z))
print(z.all_weights)