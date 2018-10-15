#!/usr/bin/env python3

import tensorflow as tf

import tensorlayer_mock as tl
from base_layer import *

tf.enable_eager_execution()

# Sample program
image = tf.ones([28 * 28 * 3], tf.float32)

x = InputLayer()()
y = MagicAddLayer(tf.constant(10))(x)

z = y(image)

print(z)