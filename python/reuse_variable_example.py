#!/usr/bin/env python3

import tensorflow as tf
import tensorlayer_mock as tl
from base_layer import InputLayer, MagicalDenseLayer

image = tf.placeholder(tf.float32, [None, 28 * 28 * 3])
x = InputLayer()(image)


layer1 = MagicalDenseLayer(tf.constant(10.0), 1000, "magic1")(x, train=True)

layer2 = MagicalDenseLayer(tf.constant(15.0), 1000)(
    layer1, train=True)

layer3 = MagicalDenseLayer(tf.constant(10.0), 1000, "magic1")(layer2, train=True, reuse=True)

# Keras-based reuse
class Model:
    def __init__(self, layers):
        self._layers = layers

    def __call__(self, x, train):
        for l in self._layers:
            x = l(x, train)
        return x


# Variable block reuse
def network(x, var_lists, train):
    with tl.reuse_variables(var_lists=None):
        x1 = MagicalDenseLayer(tf.constant(10.0), 1000, "magic1")(
            x, train=True)

        x2 = MagicalDenseLayer(tf.constant(15.0), 1000, "magic2")(
            x1, train=True)

        x3 = MagicalDenseLayer(tf.constant(15.0), 1000, "magic3")(
            x2, train=True)

train_net1 = network(x, var_lists=None, train=True)
validate_net2 = network(x, train_net1.all_weights, train=False)

# Variable selection
for var in tl.get_variables_with_name(x3.all_weights, "magic1/w1"):
    print(var)