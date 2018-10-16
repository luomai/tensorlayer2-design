#!/usr/bin/env python3

import tensorflow as tf

import tensorlayer_mock as tl
from base_layer import EagerPlaceholder, InputLayer, MagicAddLayer

tf.enable_eager_execution()

# Sample program
image = tf.ones([28 * 28 * 3], tf.float32)  # :: np.array

egr_plh = EagerPlaceholder()
egr_plh2 = EagerPlaceholder()

# begin define graph 1
# x = InputLayer()(egr_plh)
# y = MagicAddLayer(tf.constant(10.0))(x)
# z = MagicAddLayer(tf.constant(15.0))(y)
# end define graph 1

# begin define graph 2
x2 = InputLayer()(egr_plh2)
y2 = MagicAddLayer(tf.constant(10.0))(x2)
z2 = MagicAddLayer(tf.constant(15.0))(y2)
# end define graph 2

# this doesn't work
# val = egr_plh(image)  # == z.forward(y.forward(x.forward(image))) internally
# val2 = egr_plh2(image)  # == z.forward(y.forward(x.forward(image))) internally

# this doesn't work either!
# # val = z(image1, image2)  # == z.forward(y.forward(x.forward(image))) internally
# val2 = z2(image)  # == z.forward(y.forward(x.forward(image))) internally

# this might work
val = z({
    ph1: image1,
    ph2: image2,
})  # == z.forward(y.forward(x.forward(image))) internally

print(val)
