import abc

import numpy as np
from abc import abstractmethod
from accepts import accepts

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

        # Layer building state
        self._inputs_shape = None
        self._outputs_shape = None

        # Layer forward state
        self._input_layer = None
        self._inputs = None
        self._outputs = None

    @abstractmethod
    def build(self, inputs_shape):
        raise Exception(
            "The build_weights method must be implemented by inherited class"
        )

    @abstractmethod
    def forward(self, inputs, is_train):
        raise Exception(
            "The forward method must be implemented by inherited class"
        )

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

    def _add_weight(self, scope_name, var_name, shape,
                    init=np.random.normal, init_args=None):
        weight = tl.get_variable_with_initializer(
            scope_name=scope_name, var_name=var_name, shape=shape,
            init=init, init_args=init_args)
        self._weights.append(weight)  # Add into the weight collection
        self.__setattr__(var_name, weight)
        return weight

    def __call__(self, input_layer):
        # FIXME: use "*args and **kwargs" for input parameters

        if self._built:
            raise Exception(
                "The layer has been built before."
            )

        if not isinstance(input_layer, (BaseLayer, Input)):
            raise TypeError(
                "The input_layer is supposed to be a layer but got %s"
                % type(input_layer)
            )

        self._input_layer = input_layer
        self._inputs_shape = self._input_layer._outputs_shape

        self._weights = list()
        self._outputs_shape = self.build(self._inputs_shape)
        self._built = True

        return self

class Dense(BaseLayer):
    def __init__(self, n_units, act=tf.identity, name="dense"):
        super().__init__(name)
        self._n_units = n_units
        self._act = act
        # TODO: check input type

    def build(self, inputs_shape):
        if len(inputs_shape) != 2:
            raise Exception(
                "The inputs_shape of a dense layer is supposed to have 2 dims but got %s"
                % str(len(inputs_shape))
            )
        shape = [inputs_shape[1], self._n_units]
        self._add_weight(self.name, "w1", tuple(shape))
        self._add_weight(self.name, "b1", int(self._n_units))
        outputs_shape = [inputs_shape[0], self._n_units]
        return outputs_shape

    def forward(self, inputs, is_train):
        y = tf.matmul(inputs, self.w1)
        z = tf.add(y, self.b1)
        z = self._act(z)
        return z


class Dropout(BaseLayer):
    def __init__(self, keep, seed, name="dropout"):
        super().__init__(name)
        self._keep = keep
        self._seed = seed

    def build(self, inputs_shape):
        return inputs_shape

    def forward(self, inputs, is_train):
        if is_train:
            outputs = tf.nn.dropout(
                inputs,
                keep_prob=self._keep,
                seed=self._seed,
                name=self.name
            )
        else:
            outputs = inputs
        return outputs

    
class Input(BaseLayer):
    def __init__(self, inputs_shape: tuple, name="input"):
        super().__init__(name)
        # Layer constants
        self.name = name

        # Layer building state
        self._inputs_shape = inputs_shape
        self._outputs_shape = self.build(self._inputs_shape)

        # Layer forward state
        self._input_layer = None
        self._inputs = None
        self._outputs = None

    def build(self, inputs_shape):
        return inputs_shape

    def forward(self, inputs, is_train):
        return inputs
