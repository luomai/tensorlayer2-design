import abc

from abc import abstractmethod

import tensorflow as tf
import tensorlayer_mock as tl


def get_all_weights(l):
    # FIXME: remove dup
    all_weights = []

    def _visit(l):
        for layer in l.input_layer:
            if isinstance(layer, BaseLayer):
                _visit(layer)
        all_weights.extend(l.weights)

    _visit(l)
    return all_weights


# Layer API
class BaseLayer(object):
    def __init__(self, name):
        # Layer constants
        self.name = name

        # Layer weight state
        self._built = False
        self._weights = None

        # Layer forward state
        self._input_layer = None
        self._outputs = None
        self._inputs = None

    @abstractmethod
    def build(self, inputs):
        raise Exception(
            "The build_weights method must be implemented by inherited class")

    @abstractmethod
    def forward(self, inputs, is_trian):
        raise Exception(
            "The forward method must be implemented by inherited class")

    @property
    def all_weights(self):
        return get_all_weights(self)

    @property
    def weights(self):
        return self._weights

    @property
    def outputs(self):
        return self._outputs

    @property
    def inputs(self):
        return self._inputs

    @property
    def input_layer(self):
        return self._input_layer

    def _add_weight(self, scope_name, var_name, shape, train):
        # weight = tl.get_variable(
        #     scope_name=scope_name, var_name=var_name, shape=shape, train=train)
        weight = tl.get_variable_with_initializer(
            scope_name=scope_name, var_name=var_name, shape=shape, train=train)
        self._weights.append(weight)  # Add into the weight collection
        self.__setattr__(var_name, weight)
        return weight

    def __call__(self, input_layer, train):
        # FIXME: use "*args and **kwargs" for input parameters
        # if not isinstance(input_layer, list):
        #     input_layer = [input_layer]

        self._input_layer = input_layer
        # for instance in input_layer:
            # self._inputs = []
            # for input in instance._outputs:
            #     self._inputs.append(input)
        self._inputs = self._input_layer._outputs

        if not self._built:
            self._weights = []
            self.build(self._inputs, train)
            self._built = True

        self._outputs = self.forward(self._inputs)

        return self


class MagicalDenseLayer(BaseLayer):
    def __init__(self, add_constant, n_class, name="magic_dense"):
        super().__init__(name)
        self._add_constant = add_constant
        self._n_class = n_class

    def build(self, inputs):
        shape = []
        for dim in inputs.shape[1:]:
            shape.append(int(dim))
        shape.append(int(self._n_class))
        self._add_weight(self.name, "w1", tuple(shape))

    def forward(self, inputs, is_train):
        # outputs = []
        # for input in inputs:
        y = tf.matmul(inputs, self.w1)
        z = tf.add(y, self._add_constant)
        # outputs.append(z)
        return z


class Dense(BaseLayer):
    def __init__(self, n_units, act, name="dense"):
        super().__init__(name)
        self._n_units = n_units
        self._act = act

    def build(self, inputs):
        shape = []
        for dim in inputs.shape[1:]:
            shape.append(int(dim))
        shape.append(int(self._n_units))
        self._add_weight(self.name, "w1", tuple(shape))
        self._add_weight(self.name, "b1", int(self._n_units))

    def forward(self, inputs, is_train):
        # outputs = []
        # for input in inputs:
        y = tf.matmul(inputs, self.w1)
        z = tf.add(y, self.b1)
        z = self._act(z)
        # outputs.append(z)
        return z


class Dropout(BaseLayer):
    def __init__(self, keep, seed, name="dropout"):
        super().__init__(name)
        self._keep = keep
        self._seed = seed

    def build(self, inputs):
        pass
    
    def forward(self, inputs, is_train):
        if is_train:
            outputs = tf.nn.dropout(inputs, keep=self._keep, seed=self._seed, name=self.name)
        else:
            outputs = inputs
        return outputs

    
class Input(BaseLayer):
    def __init__(self, name="input"):
        super().__init__(name)

    def build(self, inputs):
        pass

    def forward(self, inputs, is_train):
        return inputs

    def __call__(self, input_tensor):

        if not self._built:
            self._weights = []
            self._built = True

        self._weights = []
        self._input_layer = None
        self._inputs = None

        # self._outputs = [input_tensor]
        self._outputs = input_tensor
        return self
