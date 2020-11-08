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
        self.zmiana_wag = []
        self.zmiana_biasow = []
        self.pobudzenia = []
        self.wyjscia = []
        self.wejscia = []
        self.errors = []
        self.wspolczynnik_uczenia = learning_rate
        for l in range(len(_layers_size) - 1):  # liczba macierzy
            self.weights.append(
                np.array([[(self.get_random_from_range(_weight_range)) for col in range(_layers_size[l])] for row in
                          range(_layers_size[l + 1])]))
            self.biases.append(self.get_random_from_range(_weight_range))
            self.pobudzenia.append(np.array([]))
            self.wyjscia.append(np.array([]))
            self.wejscia.append(np.array([]))
            self.errors.append(np.array([]))
            self.zmiana_wag.append(np.zeros((_layers_size[l+1],_layers_size[l])))
            self.zmiana_biasow.append(np.zeros(_layers_size[l+1]))
    def display_mlp(self):
        for i in self.weights:
            print('-------------')
            for j in i:
                print(j)

    def display_biases(self):
        for i in self.biases:
            print(i)

    def get_random_from_range(self, start_end):
        return (random.random() * (start_end[1] - start_end[0]) + start_end[0])

    def activation_function(self, pobudzenie, layer):
        ret = []
        if (self.activation_functions[layer] == 1):
            ret.append(self.activation_linear(pobudzenie))
        elif (self.activation_functions[layer] == 2):
            ret.append(self.activation_sigmoid(pobudzenie))
        elif (self.activation_functions[layer] == 3):
            ret.append(self.activation_tanh(pobudzenie))
        elif (self.activation_functions[layer] == 4):
            ret.append(self.activation_relu(pobudzenie))
        return np.array(ret).flatten()

    def derivative(self, pobudzenie, layer):
        ret = []
        if (self.activation_functions[layer] == 1):
            ret.append(self.derivative_linear(pobudzenie))
        elif (self.activation_functions[layer] == 2):
            ret.append(self.derivative_sigmoid(pobudzenie))
        elif (self.activation_functions[layer] == 3):
            ret.append(self.derivative_tanh(pobudzenie))
        elif (self.activation_functions[layer] == 4):
            ret.append(self.derivative_relu(pobudzenie))
        return np.array(ret)

    def activation_linear(self, x):
        return x

    def derivative_linear(self, _x):
        ret = []
        for x in _x:
            ret.append(1)
        return np.array(ret)

    def activation_tanh(self, _x):
        ret = []
        for x in _x:
            ret.append((math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x)))
        return np.array(ret)

    def derivative_tanh(self, x):
        return 1 - self.activation_tanh(x) ** 2

    def activation_sigmoid(self, _x):
        ret = []
        for x in _x:
            if x > 0:
                ret.append(1 / (1 + math.exp(-x)))
            else:
                ret.append(1 - 1 / (1 + math.exp(x)))
        return np.array(ret)

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
            sum += np.exp(x/100)
        return np.exp(_x/100) / sum

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
        return np.dot(_weights, _values) + _bias

    def forward_propagation(self, _entry_values):
        self.wejscia[0] = np.array(_entry_values) / 255
        for layer_number in range(len(self.layers_size) - 2):
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
        # return self.cross_entropy(labels, self.wyjscia[len(self.layers_size) - 2])
        return (labels - self.wyjscia[len(self.layers_size) - 2])

    def loss_function_hidden_layer(self, upper_layer_error, layer):
        return (np.dot(upper_layer_error, self.weights[layer + 1] * self.derivative(self.pobudzenia[layer], layer)))

    def cross_entropy(self, p, q):
        wynik = []
        for j in range(len(p)):
            wynik.append(-p[j] * math.log(q[j]))
        return np.array(wynik)

    def errors_back_prop(self, labels):
        self.errors[len(self.layers_size) - 2] = self.loss_function_last_layer(labels)
        for i in range(len(self.layers_size) - 2):
            self.errors[len(self.layers_size) - 3 - i] = self.loss_function_hidden_layer(
                self.errors[len(self.layers_size) - 2 - i], len(self.layers_size) - 3 - i)

    def set_change(self):
        for i in range(len(self.layers_size) - 1):
            self.zmiana_wag[i] += -np.outer(self.errors[i], self.wejscia[i]) * self.wspolczynnik_uczenia
            self.zmiana_biasow[i] += -self.errors[i] * self.wspolczynnik_uczenia


    def zamiana_wag(self):
        for i in range(len(self.weights)):
            self.weights[i] -= self.zmiana_wag[i]
            self.biases[i] -= self.zmiana_biasow[i]
        for i in range(len(self.zmiana_wag)):
            self.zmiana_biasow[i] *= 0
            self.zmiana_wag[i] *= 0

    def gradient_descent(self, wejscia, etykiety):
        for i in range(len(wejscia)):
            self.uczenie(wejscia[i], etykiety[i])
        self.zamiana_wag()


    def uczenie(self, wejscie, _etykieta):
        etykieta = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        etykieta[_etykieta] = 1
        self.forward_propagation(wejscie)
        self.errors_back_prop(etykieta)
        self.set_change()

    def uczenie_calosc(self, wejscia, etykiety, epoki, batch_size):
        for epoka in range(epoki):
            minibatch_etykiety = etykiety[epoka*batch_size:(epoka+1)*batch_size]
            minibach_wejscia = wejscia[epoka*batch_size:(epoka+1)*batch_size]
            for i in range(epoki):
                self.gradient_descent(minibach_wejscia, minibatch_etykiety)

    def wynikowa_etykieta(self, etykiety):
        max = 0
        i = 0
        for j in range(len(etykiety)):
            if (etykiety[j] > max):
                i = j
                max = etykiety[j]
        return i

    def validate(self, wejscia, etykiety):
        wszystkie = 0
        dobre = 0
        for i in range(len(wejscia)):
            wynik = self.wynikowa_etykieta(self.forward_propagation(wejscia[i]))
            print(wynik)
            if (wynik == etykiety[i]):
                wszystkie += 1
                dobre += 1
            else:
                wszystkie += 1
        return dobre, wszystkie
