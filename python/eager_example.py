#!/usr/bin/env python3

import tensorflow as tf

import tensorlayer_mock as tl
from base_layer import InputLayer, MagicalDenseLayer

tf.enable_eager_execution()

# Sample program
image = tf.ones([1, 28 * 28 * 3], tf.float32)  # :: np.array

x = InputLayer()(image)
print(x.outputs)

y = MagicalDenseLayer(tf.constant(10.0), 1000)(x, train=True, reuse=False)
# print(y.outputs)

z = MagicalDenseLayer(tf.constant(15.0), 1000)(y, train=True, reuse=False)
print(z.outputs)

exit()
