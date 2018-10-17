import abc

from abc import abstractmethod

import tensorflow as tf
# import tensorlayer as tl
import tensorlayer_mock as tl
import numpy as np

def get_all_weights(l):
    # FIXME: remove dup
    all_weights = []

    def _visit(l):
        for instance in l.input_instances:
            if isinstance(instance, BaseLayerInstance):
                _visit(instance)
        all_weights.extend(l.weights)

    _visit(l)
    return all_weights


# Layer factory API
class BaseLayerInstance(object):
    def __init__(self, input_instances):
        self._input_instances = input_instances
        self._weights = []
        self._outputs = None

        self._inputs = []
        for instance in input_instances:
            for input in instance._outputs:
                self._inputs.append(input)

    def add_attribute(self, name, attr):
        setattr(self, name, attr)

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


class EagerLayerInstance(BaseLayerInstance):
    def __init__(self, input_instances):
        super().__init__(input_instances)

class LazyLayerInstance(BaseLayerInstance):
    def __init__(self, input_instances):
        super().__init__(input_instances)

    @property
    def input_instances(self):
        return self._input_instances

    def __str__(self):
        input_spec = ','.join('?' for x in self._input_instances)  # FIXME
        output_spec = ','.join(str(y.shape) for y in self._outputs)
        return '{%s} -> {%s}, %d weights' % (input_spec, output_spec,
                                             len(self._weights))


# Layer API
class BaseLayer(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def build(self, instance, train, reuse):
        raise Exception(
            "The build_weights method must be implemented by inherited class")

    @abstractmethod
    def forward(self, instance):
        raise Exception(
            "The forward method must be implemented by inherited class")

    # Protected method
    @classmethod
    def _add_weight(cls, instance, name, shape, train, reuse):
        weight = tl.get_variable(name, shape, train, reuse)  # tl.get_variable should follow tf.keras.layers
        instance.weights.append(weight)  # Add into the weight collection
        instance.add_attribute(name, weight)
        return weight

    def __call__(self, input_instances, train, reuse):
        # FIXME: use "*args and **kwargs" for input parameters
        if not isinstance(input_instances, list):
            input_instances = [input_instances]

        if tf.executing_eagerly():
            instance = EagerLayerInstance(input_instances)
        else:
            instance = LazyLayerInstance(input_instances)

        self.build(instance, train, reuse)

        instance._outputs = self.forward(instance)

        return instance


class MagicalDenseLayer(BaseLayer):
    def __init__(self, add_constant, n_class):
        super().__init__()
        self.add_constant = add_constant
        self.n_class = n_class

    def build(self, instance, train, reuse):
        bs, n = instance.inputs[0].shape
        self._add_weight(instance, 'magic_add_weight', (int(n), self.n_class), train, reuse)

    def forward(self, instance):
        outputs = []
        for input in instance.inputs:
            y = tf.matmul(input, instance.magic_add_weight)
            z = tf.add(y, self.add_constant)
            outputs.append(z)
        return outputs


class InputLayer(BaseLayer):
    def build(self, instance):
        pass

    def forward(self, instance):
        return instance.inputs()

    def __call__(self, input_tensor):
        if tf.executing_eagerly():
            instance = EagerLayerInstance(input_instances=[])
        else:
            instance = LazyLayerInstance(input_instances=[])
        instance._outputs = [input_tensor]
        return instance