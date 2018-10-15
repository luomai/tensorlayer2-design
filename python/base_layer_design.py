import abc

from abc import abstractmethod

import tensorflow as tf
# import tensorlayer as tl
import tensorlayer_mock as tl


def overrides(f):
    return f

class BaseLayer(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def build(self, _my_layer_instance):
        raise Exception(
            "The build_weights method must be implemented by inherited class")

    @abstractmethod
    def forward(self, _my_layer_instance):
        raise Exception(
            "The forward method must be implemented by inherited class")

    # Protected method
    @classmethod
    def _add_weight(cls, instance, name, shape):
        weight = tl.get_variable(
            name, shape)  # tl.get_variable should follow tf.keras.layers
        instance.weights.append(weight)  # Add into the weight collection
        instance.add_attribute(name,
                               weight)  # Add an attribute for easy access
        return weight

    def __init__(self):
        pass

    def __call__(self, *input_layer_instances):
        instance = LayerInstance(list(input_layer_instances))
        self.build(instance)
        output_tensors = self.forward(instance)
        instance._output_tensors = output_tensors
        return instance


# physical objects
class LayerInstance(object):
    def __init__(self, inputs: [AbstractLayer]):
        self._inputs = inputs
        self._weights = []
        self._output_tensors = []

    def add_attribute(self, name, attr):
        setattr(self, name, attr)

    @property
    def all_weights(self):
        # visitor = AllWeightsVisitor()
        # return visitor.all_weights(self)
        return get_all_weights(self)

    @property
    def inputs(self):
        return self._inputs

    @property
    def weights(self):
        return self._weights

    @property
    def output_tensors(self):
        return self._output_tensors

    def __str__(self):
        input_spec = ','.join('?' for x in self._inputs)  # FIXME
        output_spec = ','.join(str(y.shape) for y in self._output_tensors)
        return '{%s} -> {%s}, %d weights' % (input_spec, output_spec,
                                             len(self._weights))


def get_all_weights(l):
    # FIXME: remove dup
    weights = []

    def _visit(l):
        for prev in l.inputs:
            if isinstance(prev, LayerInstance):
                _visit(prev)
        weights.extend(l.weights)

    _visit(l)
    return weights


class MagicAddLayer(BaseLayer):
    def __init__(self, add_constant):
        super().__init__()
        self.add_constant = add_constant
        self.m = 1000

    # @overrides(BaseLayer)
    def build(self, my_layer_instance):
        assert len(my_layer_instance.inputs) == 1
        x = my_layer_instance.inputs[0].output_tensors[0]
        _batch_size, n = x.shape
        weight = self._add_weight(my_layer_instance, 'magic_add_weight',
                                  (int(n), self.m))

    # @overrides(BaseLayer)
    def forward(self, my_layer_instance):
        assert len(my_layer_instance.inputs) == 1
        x = my_layer_instance.inputs[0].output_tensors[0]
        y = tf.matmul(x, my_layer_instance.magic_add_weight)
        output = tf.add(y, self.add_constant)
        return [output]


class InputLayer(BaseLayer):
    def build(self, my_layer_instance):
        pass

    def forward(self, my_layer_instance):
        return my_layer_instance.inputs
