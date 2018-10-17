import tensorflow as tf

from base_layer import InputLayer, MagicalDenseLayer


def simple_example(image):
    x = InputLayer()(image)
    print(x.outputs)

    y = MagicalDenseLayer(tf.constant(10.0), 1000)(x, train=True, reuse=False)
    # print(y.outputs)

    z = MagicalDenseLayer(tf.constant(15.0), 1000)(y, train=True, reuse=False)
    print(z.outputs)


def sequential_example(image):
    layers = [
        MagicalDenseLayer(tf.constant(10.0), 100),
        MagicalDenseLayer(tf.constant(15.0), 100),
    ]

    y = InputLayer()(image)

    for layer in layers:
        y = layer(y, train=True, reuse=False)

    print(y.outputs)
