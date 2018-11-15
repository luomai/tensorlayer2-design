import numpy as np
import tensorflow as tf


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


if __name__ == "__main__":
    tf.enable_eager_execution()

    weight = get_variable(
        scope_name="test", var_name="zero", shape=(3, 2), train=True)
    print(weight)
    weight = get_variable_with_initializer(
        scope_name="test", var_name="rand", shape=(3, 2), train=True)
    print(weight)
