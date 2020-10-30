import math
import random
import numpy as np


class mlp:
    def __init__(self, _layers_size, _weight_range, _bias_range, _activation_functions, _break_rule):
        self.layers_size = _layers_size #rozmiary warstw, rozmiar wejśicia podawany jest jako zerowa warstwa
        self.activation_functions = _activation_functions
        self.break_rule = _break_rule
        self.weights = []
        self.biases = []
        self.pobudzenia = []
        self.wyjscia = []
        for l in range(len(_layers_size)-1):#liczba macierzy
            self.weights.append([[(self.get_random_from_range(_weight_range))for col in range(_layers_size[l])] for row in range(_layers_size[l+1])])
            self.biases.append(self.get_random_from_range(_weight_range))
            self.pobudzenia.append([])
            self.wyjscia.append([])

    def display_mlp(self):
       for i in self.weights:
           print ('-------------')
           for j in i:
               print(j)

    def get_random_from_range(self, start_end):
        return (random.random() * (start_end[1]-start_end[0]) + start_end[0])

    def softMax(self, _x):
        x = np.array(_x)
        return x / sum(x)

    def activation_function(self, pobudzenie, layer):
        ret = []
        for i in pobudzenie:
            if(self.activation_functions[layer]==1):
                ret.append(self.activation_linear(i))
            elif(self.activation_functions[layer]==2):
                ret.append(self.activation_sigmoid(i))
            elif(self.activation_functions[layer]==3):
                ret.append(self.activation_binary(i))
        return ret

    def activation_linear(self, pobudzenie):
        if(pobudzenie >1):
            return 1
        elif(pobudzenie <0):
            return 0
        else:
            return pobudzenie

    def activation_sigmoid(self, x):
        return 1 / (1 + math.exp(-x))
    def activation_binary(self, pobudzenie):
        if(pobudzenie > 0):
            return 1
        else:
            return 0

    def pobudzenie(self, _weights, _values, _bias):
        return np.dot(_weights, np.transpose(_values)) + _bias

    def forward_propagation(self, _entry_values):
        entry_values = _entry_values
        for layer_number in range(len(self.layers_size)-1):
            self.pobudzenia[layer_number] = self.pobudzenie(self.weights[layer_number], entry_values, self.biases[layer_number])
            self.wyjscia[layer_number]= self.activation_function(self.pobudzenia[layer_number],layer_number)
            entry_values = self.wyjscia[layer_number]
        return self.softMax(self.wyjscia[len(self.layers_size)-2])