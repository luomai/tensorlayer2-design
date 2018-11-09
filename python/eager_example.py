#!/usr/bin/env python3

import time
import numpy as np
import tensorflow as tf

from base_layer import InputLayer, DenseLayer
import mnist_dataset

tf.enable_eager_execution()

class Net():
    def __init__(self):
        self.input = InputLayer("in")
        self.layers = [
            DenseLayer("fc1", 800, tf.nn.relu),
            DenseLayer("fc2", 400, tf.nn.relu),
            DenseLayer("fc3", 100, tf.nn.relu),
            DenseLayer("fc4", 10, tf.nn.relu)
        ]
        self.optimiser = tf.train.AdamOptimizer(
            learning_rate=0.0001
        )

    def forward(self, x, labels, train):
        with tf.GradientTape() as tape:
            y = self.input(x)
            for layer in self.layers:
                y = layer(y, train=train)
            loss = tf.reduce_mean(
                tf.losses.sparse_softmax_cross_entropy(
                    labels=labels,
                    logits=y.outputs
            ))
        grads = tape.gradient(loss, self.layers[-1].weights)
        return y, grads, loss

    def update(self, grads):
        self.optimiser.apply_gradients(
            zip(grads, self.layers[-1].weights)
        )

train_ds = mnist_dataset.train("../data/")
test_ds = mnist_dataset.test("../data/").batch(32)

network = Net()

log_interval = 1000

for epoch in range(3):

    start_time = time.time()

    # Training
    train_ds_shuffle = train_ds.shuffle(60000).batch(32)
    for (batch, (images, labels)) in enumerate(train_ds_shuffle):
        logits, grads, loss = network.forward(images, labels, train=True)

        pred = tf.nn.top_k(logits.outputs).indices[:, 0]
        accuracy = (pred.numpy() == labels.numpy()).mean()

        network.update(grads)

        if batch % log_interval == 0:
            rate = log_interval / (time.time() - start_time)
            print('Train: Epoch #%2d, Step #%4d\tLoss: %.4f (%d steps/sec)\tAcc: %.4f' %
                  (epoch, batch, loss, rate, accuracy))
            start_time = time.time()
    print('Train: Epoch #%2d, Step #%4d\tLoss: %.4f (%d steps/sec)\tAcc: %.4f' %
          (epoch, batch, loss, rate, accuracy))

    # Testing
    overall_accuracy = 0.0
    for (batch, (images, labels)) in enumerate(test_ds):
        logits, grads, loss = network.forward(images, labels, train=True)
        pred = tf.nn.top_k(logits.outputs).indices[:, 0]

        accuracy = (pred.numpy() == labels.numpy()).mean()
        overall_accuracy += accuracy

        if batch % log_interval == 0:
            rate = log_interval / (time.time() - start_time)
            print('Test:  Epoch #%2d, Step #%4d\tLoss: %.4f (%d steps/sec)\tAcc: %.4f' %
                  (epoch, batch, loss, rate, overall_accuracy / (batch + 1)))
    print('Test:  Epoch #%2d, Step #%4d\tLoss: %.4f (%d steps/sec)\tAcc: %.4f' %
          (epoch, batch, loss, rate, overall_accuracy / (batch + 1)))

