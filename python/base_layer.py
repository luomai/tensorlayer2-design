import abc

from abc import abstractmethod

import tensorflow as tf
from tensorflow.python.eager import context
import tensorlayer_mock as tl


def get_all_weights(l):
    # FIXME: remove dup
    weights = []

    def _visit(l):
        for prev in l._input_instances:
            if isinstance(prev, LayerInstance):
                _visit(prev)
        weights.extend(l.weights)

    _visit(l)
    return weights


class BaseLayer(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def instantiate(self, instance):
        raise Exception(
            "The build_weights method must be implemented by inherited class")

    @abstractmethod
    def forward(self, instance, inputs):
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
        input_instances = list(input_layer_instances)
        instance = LayerInstance(self, input_instances)
        self.instantiate(instance)

        # In graph mode, the LayerInstance.__call__() would not be called, so we have to call the forward
        # to initialize the tf.graph here.
        if not context.in_eager_mode():
            instance.output_tensors = []
            for input_instance in input_instances:
                outputs = input_instance(input_instance.output_tensors)
                instance.output_tensors.append(outputs)

        return instance


# physical objects
class LayerInstance(object):
    def __init__(self, layer, inputs):
        self._input_instances = inputs
        self._weights = []
        self._layer = layer

        self.output_tensors = None # This is assigned only in graph mode

    def add_attribute(self, name, attr):
        setattr(self, name, attr)

    @property
    def all_weights(self):
        # visitor = AllWeightsVisitor()
        # return visitor.all_weights(self)
        return get_all_weights(self)

    @property
    def input_instances(self):
        return self._input_instances

    @property
    def weights(self):
        return self._weights

    # @property
    # def output_tensors(self):
    #     return self._output_tensors
    #
    # def __str__(self):
    #     input_spec = ','.join('?' for x in self._input_instances)  # FIXME
    #     output_spec = ','.join(str(y.shape) for y in self._output_tensors)
    #     return '{%s} -> {%s}, %d weights' % (input_spec, output_spec,
    #                                          len(self._weights))

    def __call__(self, inputs):
        return self._layer.forward(inputs)


class MagicAddLayer(BaseLayer):
    def __init__(self, add_constant):
        super().__init__()
        self.add_constant = add_constant
        self.m = 1000

    # @overrides(BaseLayer)
    def instantiate(self, my_layer_instance):
        x = my_layer_instance.inputs[0].output_tensors[0]
        _batch_size, n = x.shape
        weight = self._add_weight(my_layer_instance, 'magic_add_weight',
                                  (int(n), self.m))

    # @overrides(BaseLayer)
    def forward(self, instance, inputs):
        outputs = []
        for input in inputs:
            y = tf.matmul(input, instance.magic_add_weight)
            output = tf.add(y, self.add_constant)
            outputs.append(output)
        return outputs


class InputLayer(BaseLayer):
    def instantiate(self, instance):
        pass

    def forward(self, instance, inputs):
        return inputs
