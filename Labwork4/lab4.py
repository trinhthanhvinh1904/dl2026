import random
import math

class Neuron:
    def __init__(self, input_size):
        self.weights = [0] * input_size
        self.bias = 0
    def random_initialize(self):
        self.weights = [
            random.random()
            for _ in range(len(self.weights))
        ]
        self.bias = random.random()
    def set_values(self, weights, bias):
        self.weights = weights
        self.bias = bias
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))
    def forward(self, inputs):
        z = self.bias
        for i in range(len(self.weights)):
            z += self.weights[i] * inputs[i]
        return self.sigmoid(z)

class Layer:
    def __init__(self, neuron_count, input_size):
        self.neurons = []
        for _ in range(neuron_count):
            neuron = Neuron(input_size)
            self.neurons.append(neuron)
    def forward(self, inputs):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.forward(inputs))
        return outputs

class NeuralNetwork:
    def __init__(self, structure_file):
        self.layers = []
        self.structure = []
        self.load_structure(structure_file)
        self.build_network()
    def load_structure(self, filename):
        with open(filename, "r") as file:
            lines = file.readlines()
        number_of_layers = int(lines[0])
        for i in range(1, number_of_layers + 1):
            self.structure.append(int(lines[i]))
    def build_network(self):
        for i in range(1, len(self.structure)):
            input_size = self.structure[i - 1]
            neuron_count = self.structure[i]
            layer = Layer(neuron_count, input_size)
            self.layers.append(layer)
    def initialize_random(self):
        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.random_initialize()
    def initialize_from_file(self, filename):
        with open(filename, "r") as file:
            lines = file.readlines()
        index = 0
        for layer in self.layers:
            for neuron in layer.neurons:
                weights = list(
                    map(float, lines[index].split())
                )
                index += 1
                bias = float(lines[index])
                index += 1
                neuron.set_values(weights, bias)
    def feedforward(self, inputs):
        outputs = inputs
        for layer in self.layers:
            outputs = layer.forward(outputs)
        return outputs

network = NeuralNetwork("struc.txt")
# network.initialize_random()
network.initialize_from_file("weight.txt")
xor_inputs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]
for x in xor_inputs:
    output = network.feedforward(x)
    print(f"{x} -> {output[0]}")