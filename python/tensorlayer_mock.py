import tensorflow as tf


def get_variable(name, shape, train, reuse):
    return tf.Variable(tf.zeros(shape), name)
