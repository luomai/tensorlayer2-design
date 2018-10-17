import tensorflow as tf


def get_variable(scope_name, var_name, shape, train, reuse):
    # TODO: Reference the tf.keras.layers
    if tf.executing_eagerly():
        var = tf.Variable(initial_value=tf.zeros(shape), name=var_name)
    else:
        with tf.variable_scope(scope_name, reuse=reuse):
            var = tf.get_variable(name=var_name, initializer=tf.zeros(shape))
    return var
