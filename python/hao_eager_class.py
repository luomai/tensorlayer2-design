#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorlayer_mock as tl
import numpy as np
from base_layer import Input, Dropout, Dense
tf.enable_eager_execution()

class MyModel(tl.Model):
    def __init__(self):
        # FIXME: currently the base tl.Model seems not really necessary?
        super(MyModel, self).__init__()

        self.input_layer = Input((None, 100))
        self.hidden_layers = [
            Dense(n_units=32, act=tf.nn.relu),
            Dense(n_units=16, act=tf.nn.relu)
        ]
        self.output_layer0 = Dense(n_units=5)
        self.output_layer1 = Dense(n_units=1)

    def __call__(self, inputs, is_train):
        # FIXME: this if looks stupid
        # check if the network has been built
        if self._inputs is None:
            # Create network
            net = self.input_layer
            for hl in self.hidden_layers:
                net = hl(net)
            net0 = self.output_layer0(net)
            net1 = self.output_layer1(net)
            self._inputs = self.input_layer
            self._outputs = [net0, net1]
        # Forward
        return super(MyModel, self).__call__(inputs, is_train)


M = MyModel()

inputs = np.ones((10, 100), dtype="float32")
outputs_train = M(inputs, True)
outputs_test = M(inputs, False)
print(outputs_train)
print(outputs_test)
