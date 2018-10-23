import tensorflow as tf

from base_layer import InputLayer, MagicalDenseLayer


def simple_example(image):
    x = InputLayer()(image)
    print(x.outputs)

    y = MagicalDenseLayer("magic1", tf.constant(10.0), 1000)(
        x, train=True)
    # print(y.outputs)

    z = MagicalDenseLayer("magic2", tf.constant(15.0), 1000)(
        y, train=True)
    print(z.outputs)
    print(z.all_weights)


def sequential_example(image):
    layers = [
        MagicalDenseLayer("magic3", tf.constant(10.0), 100),
        MagicalDenseLayer("magic4", tf.constant(15.0), 100),
    ]

    y = InputLayer()(image)

    for layer in layers:
        y = layer(y, train=True)

    print(y.outputs)
