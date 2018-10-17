#!/usr/bin/env python3

import tensorflow as tf

import tensorlayer_mock as tl
from base_layer import InputLayer, MagicalDenseLayer
from common_examples import simple_example, sequential_example

# print('x :: %s' % (x))
image = tf.placeholder(tf.float32, [None, 28 * 28 * 3])

simple_example(image)
sequential_example(image)
