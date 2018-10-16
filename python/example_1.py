#!/usr/bin/env python3

import tensorflow as tf

import tensorlayer_mock as tl
from base_layer import InputLayer, MagicAddLayer

# Sample program
image = tf.placeholder(tf.float32, [None, 28 * 28 * 3])

x = InputLayer()(image)
# print('x :: %s' % (x))

y = MagicAddLayer(tf.constant(10.0))(x)

print('y :: %s' % (y))
print(y.all_weights)

z = MagicAddLayer(tf.constant(15.0))(y)
print('z :: %s' % (z))
print(z.all_weights)