import random
import math

class Neuron:
    def __init__(self, input_size):
        self.weights = [0] * input_size
        self.bias = 0
        self.last_input = []
        self.last_output = 0
        self.last_z = 0
        self.delta = 0
    def random_initialize(self):
        self.weights = [random.random() for _ in range(len(self.weights))]
        self.bias = random.random()
    def set_values(self, weights, bias):
        self.weights = weights
        self.bias = bias
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))
    def forward(self, inputs):
        self.last_input = inputs
        z = self.bias
        for i in range(len(self.weights)):
            z += self.weights[i] * inputs[i]
        self.last_z = z
        self.last_output = self.sigmoid(z)
        return self.last_output

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
                weights = list(map(float, lines[index].split()))
                index += 1
                bias = float(lines[index])
                index += 1
                neuron.set_values(weights, bias)
    def feedforward(self, inputs):
        outputs = inputs
        for layer in self.layers:
            outputs = layer.forward(outputs)
        return outputs
    def compute_loss(self, x_data, y_data):
        total = 0.0
        n = len(x_data)
        for x, y in zip(x_data, y_data):
            output = self.feedforward(x)[0]
            z = self.layers[-1].neurons[0].last_z
            total += y * z - math.log(1 + math.exp(z))
        return -total/n
    def compute_deltas(self, y_true):
        output_layer = self.layers[-1]
        for j, neuron in enumerate(output_layer.neurons):
            s = neuron.sigmoid(-neuron.last_z)
            neuron.delta = -y_true[j] + (1 - s)
        for i in range(len(self.layers) - 2, -1, -1):
            current_layer = self.layers[i]
            next_layer = self.layers[i + 1]
            for j, neuron in enumerate(current_layer.neurons):
                s = neuron.sigmoid(-neuron.last_z)
                grad_sum = 0.0
                for next_neuron in next_layer.neurons:
                    grad_sum += next_neuron.weights[j] * next_neuron.delta
                neuron.delta = (1 - s) * s * grad_sum
    def gradient_descent(self, x_data, y_data, l, min_delta=1e-9, max_epochs=50000):
        step = 0
        fx_val_old = 0
        print("Step | f(x)")
        while True:
            fx_val = self.compute_loss(x_data, y_data)
            print(f"{step} | {fx_val}")
            if step > 0 and abs(fx_val_old - fx_val) < min_delta:
                break
            if step >= max_epochs:
                break
            fx_val_old = fx_val
            for x, y in zip(x_data, y_data):
                self.feedforward(x)
                self.compute_deltas([y])
                for layer in self.layers:
                    for neuron in layer.neurons:
                        for k in range(len(neuron.weights)):
                            neuron.weights[k] = neuron.weights[k] - l * neuron.delta * neuron.last_input[k]
                        neuron.bias = neuron.bias - l * neuron.delta
            step += 1

xor_inputs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]
xor_labels = [0, 1, 1, 0]
network = NeuralNetwork("struc.txt")
network.initialize_random()
network.gradient_descent(xor_inputs, xor_labels, l=0.5)
for x, y in zip(xor_inputs, xor_labels):
    output = network.feedforward(x)
    print(f"{x} -> predicted: {output[0]}, expected: {y}")
