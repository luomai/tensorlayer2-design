import abc

from abc import abstractmethod

import tensorflow as tf
import tensorlayer_mock as tl


def get_all_weights(l):
    # FIXME: remove dup
    all_weights = []

    def _visit(l):
        for layer in l.input_layers:
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
        self._weights = []

        # Layer forward state
        self._input_layers = None
        self._outputs = None
        self._inputs = []

    @abstractmethod
    def build(self, inputs, train):
        raise Exception(
            "The build_weights method must be implemented by inherited class")

    @abstractmethod
    def forward(self, inputs):
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
    def input_layers(self):
        return self._input_layers

    def _add_weight(self, scope_name, var_name, shape, train):
        weight = tl.get_variable(
            scope_name=scope_name,
            var_name=var_name,
            shape=shape,
            train=train)
        self._weights.append(weight)  # Add into the weight collection
        self.__setattr__(var_name, weight)
        return weight

    def __call__(self, input_layers, train):
        # FIXME: use "*args and **kwargs" for input parameters
        if not isinstance(input_layers, list):
            input_layers = [input_layers]

        self._input_layers = input_layers
        for instance in input_layers:
            for input in instance._outputs:
                self._inputs.append(input)

        if not self._built:
            self.build(self._inputs, train)
            self._built = True

        self._outputs = self.forward(self._inputs)

        return self


class MagicalDenseLayer(BaseLayer):
    def __init__(self, name, add_constant, n_class):
        super().__init__(name)
        self._add_constant = add_constant
        self._n_class = n_class

    def build(self, inputs, train):
        shape = []
        for dim in inputs[0].shape[1:]:
            shape.append(int(dim))
        shape.append(int(self._n_class))
        self._add_weight(self.name, "w1", tuple(shape), train)

    def forward(self, inputs):
        outputs = []
        for input in inputs:
            y = tf.matmul(input, self.w1)
            z = tf.add(y, self._add_constant)
            outputs.append(z)
        return outputs


class InputLayer(BaseLayer):
    def __init__(self, name="input"):
        super().__init__(name)

    def build(self, inputs, train):
        pass

    def forward(self, inputs):
        return inputs

    def __call__(self, input_tensor):
        self._outputs = [input_tensor]
        return self
