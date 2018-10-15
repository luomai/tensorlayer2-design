
import tensorflow as tf
import tensorlayer as tl

# Layer API
class BaseLayer(object):
    def __init__(self):

    @abstractmethod
    def build(self, my_layer_instance):
        raise Exception("The build_weights method must be implemented by inherited class")

    @abstractmethod
    def forward(self, my_layer_instance):
        raise Exception("The forward method must be implemented by inherited class")

    # Protected method
    def _add_weight(instance, name, shape):
        weight = tl.get_variable(name, shape) # tl.get_variable should follow tf.keras.layers
        instance.weights.append(weight) # Add into the weight collection
        instance.add_attribute('magic_add_weight', weight) # Add an attribute for easy access
        return weight

    def __call__(self, input_layer_instances):
        instance = LayerInstance()
        instance.input_layer_instances = input_layer_instances
        self.build(instance)
        instance.outputs = self.forward(instance)
        return instance
   

class MagicAddLayer(BaseLayer):
    def __init__(self, add_constant):
        super().__init__()
        self.add_constant = add_constant

    @overrides(BaseLayer)
    def build(self, my_layer_instance):
        assert my_layer_instance.input_layer_instances != None and len(my_layer_instance.input_layer_instances) == 1
        input = my_layer_instance.input_layer_instances[0] # Assume one input for simplicity
        weight = self._add_weight(my_layer_instance, 'magic_add_weight', input.shape)
        assert weight in my_layer_instance.weights and my_layer_instance.magic_add_weight is not None

    @overrides(BaseLayer)
    def forward(self, my_layer_instance):
        assert my_layer_instance.input_layer_instances != None and len(my_layer_instance.input_layer_instances) == 1
        input = my_layer_instance.input_layer_instances[0].output_tensor # Assume one input for simplicity
        x = tf.mul(input, my_layer_instance.magic_add_weight)
        output = tf.add(x, self.add_constant)
        return [output]


class LayerInstance(object):
    def __init__(self);
        self.input_layer_instances = None
        self.weights = []
        self.output_tensor = None

    def add_attribute(self, name, attr):
        setattr(self, name, attr)

    @property
    def all_weights(self):
        visitor = AllWeightsVisitor()
        return visitor.all_weights(self)
    
    @property
    def weights(self):
        return self.weights


# Network API
class BaseVisitor(object):
    def __init__(self):
        self.visited_nodes = []

    def visit(self, layer_instance):
        raise Exception("The visit method must be implemented by inherited class")

    def traverse(self, root_layer_instance):
        if root_layer_instance in self.visited_nodes:
            return
        for layer in root_layer_instance.input_layer_instances:
            self.traverse(layer)
        self.visit(root_layer_instance) # Child's visit implementation
        self.visited_nodes.append(self)

    def traverse_all(self, root_layer_instances):
        for layer in root_layer_instances:
            self.traverse(layer)


class AllWeightsVisitor(BaseVisitor):
    def __init__(self):
        super().__init__()
        self.all_weights = []

    def visit(self, layer_instance):
        self.all_weights.append(layer_instance.weights)

    def all_weights(self, root):
        self.traverse(root)
        return self.all_weights


# Sample program
input = tf.placeholder()
x = tl.layers.Input()(input)
layer = tl.layers.MagicAddLayer(tf.constant(10))(x)
print(layer.all_weights)

