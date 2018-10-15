import tensorflow as tf


def get_variable(name, shape):
    return tf.Variable(tf.zeros(shape), name)
