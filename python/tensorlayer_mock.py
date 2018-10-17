import tensorflow as tf


def get_variable(name, shape, train, reuse):
    # TODO: Reference the tf.keras.layers
    if reuse:
        return tf.get_variable(name, initializer=tf.zeros(shape))
    else:
        return tf.Variable(tf.zeros(shape), name)
