import numpy as np
import tensorflow as tf
from base_layer import BaseLayer, Input


def get_variable(scope_name, var_name, shape, train):
    # TODO: Reference the tf.keras.layers
    # if tf.executing_eagerly():
    var_name = scope_name + "/" + var_name
    var = tf.Variable(
        initial_value=tf.zeros(shape), name=var_name, trainable=train)
    # else:
    #     with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
    #         var = tf.get_variable(name=var_name, initializer=tf.zeros(shape), trainable=train)
    return var


def get_variables_with_name(all_weights, name):
    weights = []
    for weight in all_weights:
        if isinstance(weight, tf.Variable):
            if name in weight.name:
                weights.append(weight)
    return weights

def get_variable_with_initializer(scope_name, var_name, shape):
    # TODO: Reference the tf.keras.layers
    # if tf.executing_eagerly():
    var_name = scope_name + "/" + var_name
    initial_value = np.random.normal(0.0, 1.0, shape)
    var = tf.Variable(
        initial_value=tf.convert_to_tensor(initial_value, dtype=tf.float32), name=var_name)
    # else:
    #     with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
    #         var = tf.get_variable(name=var_name, initializer=tf.zeros(shape), trainable=train)
    return var


class Model():

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    def __init__(self, inputs, outputs, is_train, name="mymodel"):
        # Model properties
        self.name = name

        # Model inputs and outputs
        # TODO: check type of inputs and outputs
        self._inputs = inputs
        self._outputs = outputs

        # Model state: train or test
        self.is_train = is_train

    def __call__(self, inputs):
        # TODO: check inputs corresponds with self._inputs
        results = list()
        for out in self._outputs:
            stacked_layers = list()
            current = out
            # TODO: if inputs is not Input but BaseLayer?
            while current is not None:
                stacked_layers.append(current)
                current = current._input_layer
            # FIXME: assume there is only one inputs
            z = inputs
            for layer in stacked_layers[::-1]:
                z = layer.forward(z, self.is_train)
            results.append(z)
        return results

if __name__ == "__main__":
    pass
