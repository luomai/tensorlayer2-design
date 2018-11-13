# eager mode
def disciminator():
    inputs = tl.layers.Input([None, 100])
    net = tl.layers.Dense(n_units=32, act=tf.nn.elu)(inputs)
    net1 = tl.layers.Dense(n_units=1)(net)
    net2 = tl.layers.Dense(n_units=5)(net)
    D = tl.Model(inputs=inputs, outputs=[net1, net2])
    return D

D = disciminator()
output = D.forward(images, is_train=True)
output = D.forward(images, is_train=False)
# graph mode
def disciminator(inputs, is_train):
    net = tl.layers.Dense(n_units=32, act=tf.nn.elu)(inputs)
    net1 = tl.layers.Dense(n_units=1)(net)
    net2 = tl.layers.Dense(n_units=5)(net)
    D = tl.Model(inputs=inputs, outputs=[net1, net2], is_train=is_train)
    return D
inputs = tf.placeholder("float32", [None, 100])
D = disciminator(inputs, is_train=True)
D2 = tl.Model(reuse=True, is_train=False, model=D)
output = sess.run(D.outputs, feed_dict={inputs: images})
output = sess.run(D2.outputs, feed_dict={inputs: images})
