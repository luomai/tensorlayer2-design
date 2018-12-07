#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorlayer as tl
import numpy as np
from base_layer import Input, Dropout, Dense
tf.enable_eager_execution()

class MyModel(tl.Model):
    def __init__(self):
        self.inputs = tl.layers.Input([None, 100])
        self.hidden_layers = [
            tl.layers.Dense(n_units=32, act=tf.nn.relu)
            tl.layers.Dense(n_units=16, act=tf.nn.relu)
        ]
        self.output_layer0 = tl.layers.Dense(n_units=5)
        self.output_layer1 = tl.layers.Dense(n_units=1)
    
    def __call__(self, input, foo=0):
         net = self.inputs(input)
         for hl in self.hidden_layers:
             net = hl(net)
         if foo ==  0:
             return self.output_layer0(net)
         else
             return self.output_layer1(net)

M = MyModel()

inputs = np.ones((10, 100))
output0 = M(inputs, foo=0)
output1 = M(inputs, foo=1)
