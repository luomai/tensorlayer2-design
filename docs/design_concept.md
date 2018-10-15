# Rethinking TensorLayer 2.0

We are studying tf.keras.layers, pytorch.layers, TF eager and graph modes. We are thinking to refactor the core layer and the way to implement layer. The current implementation has two limitations. 1) it doesn’t support eager mode, we want it support both eager and graph so that the coming of TF 2.0 does not require us to refactor the layers again, and 2) the current layer implementation is more complex than tf.keras.layers, making people difficult to contribute layers to TL. The original one is very TensorFlow-friendely and the developer does not require to learn, for example, the role of self._temp_data, which loss the transparent feature.


## What Happen to TensorFlow 2.0

- Eager mode become the default mode.
- Remove `tf.get_variable`
- Remove `tf.variable_scope`
- Remove `tf.layers` (graph mode) and force users to use `tf.keras.layers` (eager mode)

TF does not long support using name scope to control the reusing. I think it is a good change to keep TL's name scope advantage which can help us to get users from `tf.layers`.

## Interface Design

### tf.Keras

```python
class ObjectDet(tf.keras.Model):
    def __init__(self):
        super(ObjectDet,self).__init__()
        self.layer1=tf.keras.layers.Conv2D(32, (3, 3), input_shape=(28,28,1), padding='same', activation='relu')
        self.layer2=tf.keras.layers.Dropout(0.2)
        self.layer3=tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')
        self.layer4=tf.keras.layers.MaxPooling2D(pool_size=(2,2))
        self.layer5=tf.keras.layers.Flatten()
        self.layer6=tf.keras.layers.Dense(512, activation='relu')
        self.layer7=tf.keras.layers.Dropout(0.1)
        self.layer8=tf.keras.layers.Dense(10, activation='softmax')

    def call(self, input):
        """Run the model."""
        result = self.layer1(input)
        result = self.layer2(result)
        result = self.layer3(result)
        result = self.layer4(result)
        result = self.layer5(result)
        result = self.layer6(result)
        result = self.layer7(result)
        result = self.layer8(result)

        return result

model = ObjectDet()
result1 = model(x1)
result2 = model(x2)
```

[tf.keras.Conv2D](https://tensorflow.google.cn/api_docs/python/tf/keras/layers/Conv2D)

    * model.variables all weights in the model
    * model.weights the weights of the latest layer
    * model.output the output tensor(s)
    * model.trainable_variables
    * model.trainable_weights
    * input, model.dtype, input_shape, output_shape, name

Keras does not use name to control reuse anymore, they implicitly create `self.layer` and use them via `call`. The advantage is that it is good for beginner to understand how to reuse the layers, but the disadvantage is that users need to write two lines of code for one layers.

### TensorLayer 2.0

To solve this disadvantage, I believe naming is good for expert users, and beginner should be able to learn it quick. As shown in the following, each layer has a unique name, and TL has a global name dictionary to store the `tf.Variable`, then allow users to reuse the variables/weights via name.

```python
def model(x, is_train=True, reuse=False):
    with tl.layers.variable_name("MLP", reuse=reuse):
        network = tl.layers.Input(name='input')(x)
        network = tl.layers.Dropout(keep=0.8, is_fix=True, name='drop1')(network, is_train=is_train)
        network = tl.layers.Dense(n_units=800, act=tf.nn.relu, name='relu1')(network)
        network = tl.layers.Dropout(keep=0.5, is_fix=True, name='drop2')(network, is_train=is_train)
        network = tl.layers.Dense(n_units=800, act=tf.nn.relu, name='relu2')(network)
        network = tl.layers.Dropout(keep=0.5, is_fix=True, name='drop3')(network, is_train=is_train)
        network = tl.layers.Dense(n_units=10, act=None, name='output')(network)
    return network

net1 = model(x, is_train=True, reuse=False)
net2 = model(x, is_train=False, reuse=True)

```

    * network.all_weights all weights in the model
    * network.local_weights the weights of the latest layer
    * network.outputs the output tensor

    * network.print_weights()
    * network.print_outputs()

## TL 1.x to TL 2.x API Changes

| TL 1.x                                     | TL 2.x                                 | Description                                |
|--------------------------------------------|----------------------------------------|--------------------------------------------|
| tl.files.assign_params()                   | tl.files.assign_weights()              | assign list of numpy array to weights      |
| net.print_params()                         | net.print_weights()                    | print all weights                          |
| net.print_layers()                         | net.print_outputs()                    | print all layers outputs                   |
| net.all_params                             | net.all_weights                        | --                                         |
| net.all_layers                             | net.all_outputs                        | --                                         |
| --                                         | net.local_weights                      | --                                         |
| net.count_params()                         | net.count_all\_weights()               | count number of weight values              |
| --                                         | net.count_local\_weights()             | count number of weigh values of this layer |
| tl.models.                                 | tl.hub                                 | --                                         |
| Vgg.restore_params()                       | Vgg.restore\_weights()                 | --                                         |
| tl.layers.get_layers_with_name()           | tl.layers.get_outputs\_with\_name()    | --                                         |
| tf.get_variable()                          | tl.get_variable()                      | --                                         |
| tf.variable_scope()                        | tl.layers.variable_name()              | --                                         |
| n1 = tl.layers.SomeLayer(n0, ...,name='xx) | n1 = tl.layers.Some(...)(n0,name='xx') | argument net moved                         |

Apart from that, we simplify all layer APIs (e.g., `DenseLayer` --> `Dense`) except professional APIs (i.e., `Conv1dLayer`, `Conv2dLayer`, `Conv3dLayer`, `DeConv2dLayer`, `DeConv3dLayer`, `PadLayer`, `PoolLayer`), and raise exception when users use the old one.

## Implementation Ideas

Store a global dictionary for TensorFlow variables in TensorFlow backend.

```python
tf.__global__.tl_variables = {''}

def get_variable(name):
    name = tl.variable_name + '/' + name
    if tf.__global__.tl_variables[name] is None:
        v = tf.Variable(....)
        tf.__global__.tl_variables[name] = v
    else:
    	 if is_reuse:
 	       v = tf.__global__.tl_variables[name]
 	     else:
 	       raise Exception("variable {} exists, do you mean tl.layers.variable_name(..., reuse=True) ?" % (name)")
    return v
```

Then we can remove `tf.get_variable()` and `tf.variable_scope`. To support Eager mode, we use `build_weights()` to create variables using `tf.Variable`.

```python
class Dense(Layer):
    def __init__(
            self,
            # prev_layer,
            n_units=100,
            act=None,
            W_init=tf.truncated_normal_initializer(stddev=0.1),
            b_init=tf.constant_initializer(value=0.0),
            W_init_args=None,
            b_init_args=None,
            name='dense',
    ):
        # 1. save act inside for `self._apply_activation`
        # 2. W(b)_init_args will be changed to empty dictionary {}
        # 3. self.name will be changed to 'MLP/dense' if using `variable_name('MLP')` outside.
        # 4. get net.inputs from prev_layer.outputs
        super(Dense, self
             ).__init__(act=act, W_init_args=W_init_args, b_init_args=b_init_args, name=name)

        if self.inputs.get_shape().ndims != 2:
            raise AssertionError("The input dimension must be rank 2, please reshape or flatten it")

        self.n_in = int(self.inputs.get_shape()[-1])
        self.n_units = n_units
        self.W_init = W_init
        self.b_init = b_init

 	 def __str__:
        logging.info(
            "Dense  %s: %d %s" %
            (self.name, n_units, self.act.__name__ if self.act is not None else 'No Activation')
        )

    def build_weights():

        with tl.layers.variable_name(self.name):
            # name of self.W will be `MLP/dense/W`
            self.W = tl.layers.get_variable(
                name='W', shape=(self.n_in, self.n_units), initializer=self.W_init, dtype=LayersConfig.tf_dtype, **self.W_init_args
            )
            if self.b_init is not None:
                try:
                    self.b = tl.layers.get_variable(
                        name='b', shape=(self.n_units), initializer=self.b_init, dtype=LayersConfig.tf_dtype, **self.b_init_args
                    )
                except Exception:  # If initializer is a constant, do not specify shape.
                    self.b = tl.layers.get_variable(name='b', initializer=b_init, dtype=LayersConfig.tf_dtype, **self.b_init_args)

        # add weights to local_weights, all_weights ...
        if self.b_init is not None:
            self._add_weights([self.W, self.b])
        else:
            self._add_weights(self.W)

    def forward(prev_layer):
        self.outputs = tf.matmul(prev_layer.outputs, self.W)
        self.outputs = tf.nn.bias_add(self.outputs, self.b, name='bias_add')
        self.outputs = self._apply_activation(self.outputs)

        self._add_outputs.append(self.outputs) # ?? is this the way to append outputs ?
```

For layers that use Keras:

```python
class Conv2d(Layer):
    def __init__(
            self,
            # prev_layer,
            n_filter=32,
            filter_size=(3, 3),
            strides=(1, 1),
            act=None,
            padding='SAME',
            data_format='channels_last',
            dilation_rate=(1, 1),
            W_init=tf.truncated_normal_initializer(stddev=0.02),
            b_init=tf.constant_initializer(value=0.0),
            W_init_args=None,
            b_init_args=None,
            use_cudnn_on_gpu=None,
            name='conv2d',
    ):
        super(Conv2d, self
             ).__init__(act=act, W_init_args=W_init_args, b_init_args=b_init_args, name=name)

        self.n_filter = n_filter
        self.filter_size = filter_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.W_init = W_init
        self.b_init = b_init

    def __str__():
        logging.info(
            "Conv2d %s: n_filter: %d filter_size: %s strides: %s pad: %s act: %s" % (
                self.name, n_filter, str(filter_size), str(strides), padding,
                self.act.__name__ if self.act is not None else 'No Activation'
            )
        )

    def build_weights():
        self.conv2d = tf.keras.layers.Conv2D(
            # inputs=self.inputs,
            filters=self.n_filter,
            kernel_size=self.filter_size,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
            activation=self.act,
            use_bias=(False if self.b_init is None else True),
            kernel_initializer=self.W_init,  # None,
            bias_initializer=self.b_init,  # f.zeros_initializer(),
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            trainable=self.is_train,
            name=self.name,
            # reuse=None,
        )
        new_variables = self.conv2d.weights

        # self._add_outputs(self.outputs)
        self._add_weights(new_variables)

    def forward(self):
        # warning, self.conv2d must put before ``new_variables = self.conv2d.weights``?
        self.outputs = self.conv2d(self.inputs)

        self._add_outputs.append(self.outputs) # ?? is this the way to append outputs ?
```

Code must be tested with `tf.enable_eager_execution()`.
