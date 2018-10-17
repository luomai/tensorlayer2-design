#!/usr/bin/env python3

import tensorflow as tf
import numpy as np

from common_examples import simple_example, sequential_example

tf.enable_eager_execution()

# Sample program
image = tf.ones([1, 28 * 28 * 3], tf.float32)  # :: np.array

simple_example(image)
sequential_example(image)

ndarray = np.ones([1, 28 * 28 * 3], dtype=np.float32)
simple_example(ndarray)
