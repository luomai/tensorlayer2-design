#!/usr/bin/env python3

import tensorflow as tf

import tensorlayer_mock as tl
from base_layer import EagerPlaceholder, InputLayer, MagicAddLayer

tf.enable_eager_execution()

# Sample program
image = tf.ones([28 * 28 * 3], tf.float32)

egr_plh = EagerPlaceholder()

x = InputLayer()(egr_plh)
y = MagicAddLayer(tf.constant(10.0))(x)
val = egr_plh(image)
print(val)

z = MagicAddLayer(tf.constant(15.0))(y)
val = egr_plh(image)
print(val)