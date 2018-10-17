#!/usr/bin/env python3

import tensorflow as tf

import tensorlayer_mock as tl
from base_layer import InputLayer, MagicalDenseLayer
from common_examples import simple_example, sequential_example

tf.enable_eager_execution()

# Sample program
image = tf.ones([1, 28 * 28 * 3], tf.float32)  # :: np.array

simple_example(image)
sequential_example(image)
