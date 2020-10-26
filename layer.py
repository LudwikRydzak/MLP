import random

import numpy as np
class layer:
    def __init__(self, _layers_size, _weight_range, _bias_range, _activation_function ):
        self.layers_size = _layers_size
        self.activation_function = _activation_function
        for j in range(len(_layers_size)):
            for i in range(_layers_size):
                self.weights[j].append(self.get_random_from_range(_weight_range))
            self.biases.append(self.get_random_from_range(_weight_range))

    def get_random_from_range(self, start_end):
        return (random.random() * (start_end[1]-start_end[0]) + start_end[0])

    def softMax(self, x):
        return x / sum(x)

    def activation_function(self):
        if(self.activation_function()==1):
            self.activation_linear()
        elif(self.activation_function()==2):
            self.activation_sigmoidal()
        elif(self.activation_function()==3):
            self.activation_binary()

    def activation_linear(self):
        return 0
    def activation_sigmoidal(self):
        return 0
    def activation_binary(self):
        return 0

    def pobudzenie(self, values):
        return np.dot(self.weights, np.transpose(values))