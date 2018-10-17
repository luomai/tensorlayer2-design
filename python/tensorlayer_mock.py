import tensorflow as tf


def get_variable(name, shape, train, reuse):
    # TODO: Reference the tf.keras.layers
    return tf.Variable(tf.zeros(shape), name)
