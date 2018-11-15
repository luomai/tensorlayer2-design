import tensorflow as tf

from base_layer import InputLayer, MagicalDenseLayer


def simple_example(image):
    x = InputLayer()(image)
    print(x.outputs)

    y = MagicalDenseLayer(tf.constant(10.0), 1000, "magic1")(x)
    # print(y.outputs)

    z = MagicalDenseLayer(tf.constant(15.0), 1000, "magic2")(y)
    print(z.outputs)
    print(z.all_weights)


def sequential_example(image):
    layers = [
        MagicalDenseLayer(tf.constant(10.0), 100, "magic3"),
        MagicalDenseLayer(tf.constant(15.0), 100, "magic4"),
    ]

    y = InputLayer()(image)

    for layer in layers:
        y = layer(y)

    print(y.outputs)
