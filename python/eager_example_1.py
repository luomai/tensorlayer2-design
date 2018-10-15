#!/usr/bin/env python3

import tensorflow as tf

tf.executing_eagerly() 

from base_layer import *

# Sample program
image = tf.ones([28 * 28 * 3], tf.float32)

x = InputLayer()(image)
print('x :: %s' % (x))

y = MagicAddLayer(tf.constant(10.0))(x)
print('y :: %s' % (y))

print(y.all_weights)
