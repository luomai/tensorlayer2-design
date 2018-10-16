import abc

from abc import abstractmethod

import tensorflow as tf
# import tensorlayer as tl
import tensorlayer_mock as tl


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


class BaseLayer(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def build(self, instance, input_shapes):
        raise Exception(
            "The build_weights method must be implemented by inherited class")

    @abstractmethod
    def forward(self, instance, inputs):
        raise Exception(
            "The forward method must be implemented by inherited class")

    # Protected method
    @classmethod
    def _add_weight(cls, instance, name, shape):
        weight = tl.get_variable(name, shape)  # tl.get_variable should follow tf.keras.layers
        instance.weights.append(weight)  # Add into the weight collection
        instance.add_attribute(name, weight)
        return weight

    def __init__(self):
        pass

    def __call__(self, *input_instances):
        if tf.executing_eagerly():
            instance = EagerLayerInstance(list(input_instances))

        else:
            instance = LazyLayerInstance(list(input_instances))

            inputs = []
            for instance in input_instances:
                for input in instance._outputs:
                    inputs.append(input)
            input_shapes = [input.shape for input in inputs] # shape[0] is batch size
            self.build(instance, input_shapes)
            instance._outputs = self.forward(instance, inputs)
            return instance


class BaseLayerInstance(object):
    def __init__(self):
        self._weights = []

    def add_attribute(self, name, attr):
        setattr(self, name, attr)

    @property
    def all_weights(self):
        return get_all_weights(self)

    @property
    def weights(self):
        return self._weights


class EagerLayerInstance(BaseLayerInstance):
    def __init__(self, output_instances, layer):
        super().__init__()
        self._output_instances = output_instances
        self._layer = layer

    def compute(self, inputs):
        return self._layer.forward(self, inputs)


class LazyLayerInstance(BaseLayerInstance):
    def __init__(self, input_instances):
        super().__init__()
        self._input_instances = input_instances
        self._outputs = []

    @property
    def input_instances(self):
        return self._input_instances

    @property
    def outputs(self):
        return self._outputs

    def __str__(self):
        input_spec = ','.join('?' for x in self._input_instances)  # FIXME
        output_spec = ','.join(str(y.shape) for y in self._outputs)
        return '{%s} -> {%s}, %d weights' % (input_spec, output_spec,
                                             len(self._weights))


class MagicAddLayer(BaseLayer):
    def __init__(self, add_constant):
        super().__init__()
        self.add_constant = add_constant
        self.m = 1000

    # @overrides(BaseLayer)
    def build(self, instance, input_shapes):
        batch_size, n = input_shapes[0] # assume only one
        self._add_weight(instance, 'magic_add_weight', (int(n), self.m))

    # @overrides(BaseLayer)
    def forward(self, instance, inputs):
        outputs = []
        for input in inputs:
            y = tf.matmul(input, instance.magic_add_weight)
            z = tf.add(y, self.add_constant)
            outputs.append(z)
        return outputs


class InputLayer(BaseLayer):
    def build(self, instance, input_shapes):
        pass

    def forward(self, instance, inputs):
        return inputs

    def __call__(self, placeholder):
        if tf.executing_eagerly():
            instance = EagerLayerInstance([], self)
            return instance
        else:
            instance = LazyLayerInstance([])
            instance._outputs = self.forward(instance, [placeholder])
            return instance


class EagerPlaceholder(object):
    def __init__(self):
        pass

    def __call__(self, input):
        pass