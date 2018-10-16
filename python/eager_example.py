#!/usr/bin/env python3

import tensorflow as tf

import tensorlayer_mock as tl
from base_layer import EagerPlaceholder, InputLayer, MagicAddLayer

tf.enable_eager_execution()

# Sample program
image1 = tf.ones([28 * 28 * 3], tf.float32)  # :: np.array
image2 = tf.ones([28 * 28 * 3], tf.float32)  # :: np.array

egr_plh1 = EagerPlaceholder()
egr_plh2 = EagerPlaceholder()

# begin define graph 1
x1 = InputLayer()(egr_plh1)
x2 = InputLayer()(egr_plh2)
y = SumLayer()(x1, x2)
z = MagicAddLayer(tf.constant(15.0))(y)
# end define graph 1

# this doesn't work
# val = egr_plh(image)  # == z.forward(y.forward(x.forward(image))) internally
# val2 = egr_plh2(image)  # == z.forward(y.forward(x.forward(image))) internally

# this doesn't work either!
# # val = z(image1, image2)  # == z.forward(y.forward(x.forward(image))) internally
# val2 = z2(image)  # == z.forward(y.forward(x.forward(image))) internally

# this might work
val = z({
    egr_plh1: image1,
    egr_plh2: image2,
})  # == z.forward(y.forward(x.forward(image))) internally

print(val)
