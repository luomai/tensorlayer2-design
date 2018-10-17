#!/usr/bin/env python3

import tensorflow as tf

import tensorlayer_mock as tl
from base_layer import InputLayer, MagicAddLayer

tf.enable_eager_execution()

# Sample program
image1 = tf.ones([28 * 28 * 3], tf.float32)  # :: np.array
image2 = tf.ones([28 * 28 * 3], tf.float32)  # :: np.array

x = InputLayer()(image)
print(x.outputs)

y = MagicAddLayer(tf.constant(10.0))(x)
print(y.outputs)
exit()

z = MagicAddLayer(tf.constant(15.0))(y)
# end define graph 1

# val = z.forward(y.forward(x.forward(image)))
print(z.outputs)
