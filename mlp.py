import math
import random
import numpy as np


class mlp:
    def __init__(self, _layers_size, _weight_range, _bias_range, _activation_functions, _break_rule, learning_rate):
        self.layers_size = _layers_size  # rozmiary warstw, rozmiar wejśicia podawany jest jako zerowa warstwa
        self.activation_functions = _activation_functions
        self.break_rule = _break_rule
        self.weights = []
        self.biases = []
        self.pobudzenia = []
        self.wyjscia = []
        self.wejscia = []
        self.errors = []
        self.wspolczynnik_uczenia = learning_rate
        for l in range(len(_layers_size) - 1):  # liczba macierzy
            self.weights.append(
                [[(self.get_random_from_range(_weight_range)) for col in range(_layers_size[l])] for row in
                 range(_layers_size[l + 1])])
            self.biases.append(self.get_random_from_range(_weight_range))
            self.pobudzenia.append([])
            self.wyjscia.append([])
            self.wejscia.append([])
            self.errors.append([])

    def display_mlp(self):
        for i in self.weights:
            print('-------------')
            for j in i:
                print(j)

    def get_random_from_range(self, start_end):
        return (random.random() * (start_end[1] - start_end[0]) + start_end[0])

    def activation_function(self, pobudzenie, layer):
        ret = []
        for i in pobudzenie:
            if (self.activation_functions[layer] == 1):
                ret.append(self.activation_linear(i))
            elif (self.activation_functions[layer] == 2):
                ret.append(self.activation_sigmoid(i))
            elif (self.activation_functions[layer] == 3):
                ret.append(self.activation_tanh(i))
            elif (self.activation_functions[layer] == 4):
                ret.append(self.activation_relu(i))
        return np.array(ret)

    def derivative(self, pobudzenie, layer):
        ret = []
        for i in pobudzenie:
            if (self.activation_functions[layer] == 1):
                ret.append(self.derivatve_linear(i))
            elif (self.activation_functions[layer] == 2):
                ret.append(self.derivative_sigmoid(i))
            elif (self.activation_functions[layer] == 3):
                ret.append(self.derivative_tanh(i))
            elif (self.activation_functions[layer] == 4):
                ret.append(self.derivative_relu(i))
        return np.array(ret)

    def activation_linear(self, x):
        return x

    def derivative_linear(self, _x):
        ret = []
        for x in _x:
            ret.append(1)
        return np.array(ret)

    def activation_tanh(self, x):
        return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))

    def derivative_tanh(self, x):
        return 1 - self.activation_tanh(x) ** 2

    def activation_sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def derivative_sigmoid(self, x):
        sig = self.activation_sigmoid(x)
        return sig * (1 - sig)

    def activation_relu(self, _x):
        ret = []
        for x in _x:
            if x > 0:
                ret.append(x)
            else:
                ret.append(0)
        return np.array(ret)

    def derivative_relu(self, _x):
        ret = []
        for x in _x:
            if x > 0:
                ret.append(1)
            else:
                ret.append(0)
        return np.array(ret)

    def softMax(self, _x):
        sum = 0
        for x in _x:
            sum += np.exp(x)
        return np.exp(_x) / sum

    def derivative_softmax(self, s):
        # https://stackoverflow.com/questions/54976533/derivative-of-softmax-function-in-python
        jacobian = np.diag(s)  # inicjuję macierz
        for i in range(len(jacobian)):
            for j in range(len(jacobian)):
                if i == j:
                    jacobian[i][j] = s[i] * (1 - s[i])  # jesli i=j, to jeden wzorek na pochodna z wykladu
                else:
                    jacobian[i][j] = -s[i] * s[j]  # jesli nie, to drugi wzorek na pochodną z wykladu
        return jacobian

    def pobudzenie(self, _weights, _values, _bias):
        return np.dot(_weights, np.transpose(_values)) + _bias

    def forward_propagation(self, _entry_values):
        self.wejscia[0] = _entry_values
        # entry_values = _entry_values
        for layer_number in range(len(self.layers_size) - 2):
            print(layer_number)
            self.pobudzenia[layer_number] = self.pobudzenie(self.weights[layer_number], self.wejscia[layer_number],
                                                            self.biases[layer_number])
            self.wyjscia[layer_number] = self.activation_function(self.pobudzenia[layer_number], layer_number)
            self.wejscia[layer_number + 1] = self.wyjscia[layer_number]
        self.pobudzenia[len(self.layers_size) - 2] = self.pobudzenie(self.weights[len(self.layers_size) - 2],
                                                                     self.wejscia[len(self.layers_size) - 2],
                                                                     self.biases[len(self.layers_size) - 2])
        self.wyjscia[len(self.layers_size) - 2] = self.softMax(
            self.pobudzenia[len(self.layers_size) - 2])  # ostatnia warstwa bez funkcji aktywacji, ostatnia ma softmax
        return self.wyjscia[len(self.layers_size) - 2]

    def loss_function_last_layer(self, labels):
        return (labels - self.wyjscia[len(self.layers_size) - 2]) * self.derivative_softmax(
            self.pobudzenia[len(self.layers_size) - 2])

    def loss_function_hidden_layer(self, upper_layer_error, layer):
        return self.derivative(self.pobudzenia[layer], layer) * np.dot(self.weights[layer], upper_layer_error)

    def cross_entropy(p, q):
        suma = 0
        for j in range(len(p)):
            suma += -p[j] * math.log(q[j])
        return suma

    def errors_back_prop(self, labels):
        self.errors[len(self.layers_size) - 2] = self.loss_function_last_layer()
        for i in len(self.layers_size) - 2:
            self.errors[len(self.layers_size) - 3 - i] = self.loss_function_hidden_layer(
                self.errors[len(self.layers_size) - 2 - i], len(self.layers_size) - 3 - i)

    def change_weights(self):
        for i in len(self.layers_size) - 1:
            self.weights[i] = self.weights[i] + self.wspolczynnik_uczenia * np.dot(self.errors[i], self.wejscia[i])

    def uczenie(self, wejscie, etykieta):
        self.forward_propagation(wejscie)
        self.errors_back_prop(etykieta)
        self.change_weights()

    def uczenie_calosc(self, wejscia, etykiety):
        for i in len(wejscia):
            self.uczenie(wejscia[i], etykiety[i])
