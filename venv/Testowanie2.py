import matplotlib.pyplot as plt
from mlp import *


class Testowanie2:
    def __init__(self):
        self.hidden_layer_sizes = [1, 2, 3, 5, 7, 10, 12, 14, 16, 18, 20, 30, 40, 50, 70, 90,
                                   100, 200, 300, 400, 500, 600, 700, 800]
        self.batch_sizes = [1, 2, 4, 8, 10, 16, 20, 30, 32, 40, 50, 60, 64, 70, 80, 90, 100, 128, 200, 256, 300, 400,
                            512]
        self.weights_ranges = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5, 10]
        # self.learning_factors = [0.01]
        self.learning_factors = [0.001, 0.01, 0.1, 0.2, 0.5, 0.8, 0.9, 0.99, 1, 2]
        self.activation_functions = [1, 2, 3, 4]
        self.break_rule = 0.1
        self.biasrange = [-1, 1]
        self.x_epoki = []
        self.x_procenty = []
        self.rozmiary_sieci = [784, 40, 10]
        self.wspolczynnik_momentum = [0.1, 0.2, 0.5, 0.8, 0.85, 0.9, 0.99]
        self.wspolczynnik_momentum2 = [0.1, 0.2, 0.5, 0.8, 0.85, 0.9, 0.99]
        self.opcja_momentum = [1, 2, 3, 4, 5]

    def f1measure(self, TP, TN, FP, FN):
        return (2 * TP) / (2 * TP + FP + FN)

    def accuracy(self, TP, TN, FP, FN):
        return (TP + TN) / (TP + TN + FP + FN)

    def precision(self, TP, TN, FP, FN):
        return TP / (TP + FP)

    def recall(self, TP, TN, FP, FN):
        return TP / (TP + FN)

    def specifity(self, TP, TN, FP, FN):
        return TN / (TN + FP)

    def testowe_przejscie(self, train_set, train_labels, test_set, test_labels):
        f1measure = 0
        accuracy = 0
        precision = 0
        recall = 0
        specifity = 0
        suma_epok = 0
        table = [0]*200
        for iteracja in range(10):
            network = mlp(self.rozmiary_sieci, [-1, 1], self.biasrange, [4], self.break_rule,
                          0.01, 1, 0, 0.999,3)
            liczba_epok , temp = network.uczenie_calosc(train_set, train_labels, epoki=200, batch_size=16)
            for i in range(200):
                table[i] = table[i] + temp[i]
            np.set_printoptions(suppress=True)
            TP, TN, FP, FN, macierz_odpowiedzi = network.validate(test_set, test_labels)
            f1measure += self.f1measure(TP, TN, FP, FN)
            accuracy += self.accuracy(TP, TN, FP, FN)
            precision += self.precision(TP, TN, FP, FN)
            recall += self.recall(TP, TN, FP, FN)
            specifity += self.specifity(TP, TN, FP, FN)
            suma_epok += liczba_epok
        print(f1measure/10)
        table = [x/10 for x in table]
        self.show_przejscie(range(200),table)

    def show_przejscie(self, x, y):
        plt.title('Średnia wartosc funkcji straty dla przykładowego uczenia')
        plt.xlabel('epoka')
        plt.ylabel('loss')
        plt.plot(x, y, 'bo', label='średnia wartosc funkcji straty')
        plt.legend()
        plt.show()

    def test_momentum(self, train_set, train_labels, test_set, test_labels):
        with open('badania_momentum_relu.txt', 'w') as file:
            file.writelines(
                'wspolczynnik_uczenia;wspolczynnik momentum;liczba epok;accuracy;precision;recall;specifity;f1measure\n')
            for wspolczynnik_momentum in self.wspolczynnik_momentum:
                for wspolczynnik_uczenia in self.learning_factors:
                    f1measure = 0
                    accuracy = 0
                    precision = 0
                    recall = 0
                    specifity = 0
                    suma_epok = 0
                    print(f'wspolczynnniki:{wspolczynnik_uczenia},{wspolczynnik_momentum}')
                    for iteracja in range(10):
                        network = mlp(self.rozmiary_sieci, [-1, 1], self.biasrange, [4], self.break_rule,
                                      wspolczynnik_uczenia, 1, wspolczynnik_momentum, self.wspolczynnik_momentum2)
                        liczba_epok = network.uczenie_calosc(train_set, train_labels, epoki=200, batch_size=16)
                        print(
                            f'iteracja: {iteracja} badana zmiana:{wspolczynnik_uczenia},{wspolczynnik_momentum} liczba epok: {liczba_epok}\n')
                        np.set_printoptions(suppress=True)
                        TP, TN, FP, FN, macierz_odpowiedzi = network.validate(test_set, test_labels)
                        f1measure += self.f1measure(TP, TN, FP, FN)
                        accuracy += self.accuracy(TP, TN, FP, FN)
                        precision += self.precision(TP, TN, FP, FN)
                        recall += self.recall(TP, TN, FP, FN)
                        specifity += self.specifity(TP, TN, FP, FN)
                        suma_epok += liczba_epok
                    file.writelines(
                        f'{wspolczynnik_uczenia};{wspolczynnik_momentum};{suma_epok / 10};{accuracy / 10};{precision / 10};{recall / 10};{specifity / 10};{f1measure / 10}\n')

    def test_momentum_nesterova(self, train_set, train_labels, test_set, test_labels):
        with open('badania_momentum_nesterova_relu.txt', 'w') as file:
            file.writelines(
                'wspolczynnik_uczenia;wspolczynnik momentum;liczba epok;accuracy;precision;recall;specifity;f1measure\n')
            for wspolczynnik_momentum in self.wspolczynnik_momentum:
                for wspolczynnik_uczenia in self.learning_factors:
                    f1measure = 0
                    accuracy = 0
                    precision = 0
                    recall = 0
                    specifity = 0
                    suma_epok = 0
                    print(f'wspolczynnniki:{wspolczynnik_uczenia},{wspolczynnik_momentum}')
                    for iteracja in range(10):
                        network = mlp(self.rozmiary_sieci, [-1, 1], self.biasrange, [4], self.break_rule,
                                      wspolczynnik_uczenia, 2,
                                      wspolczynnik_momentum, self.wspolczynnik_momentum2)
                        liczba_epok = network.uczenie_calosc(train_set, train_labels, epoki=200, batch_size=16)
                        print(
                            f'iteracja: {iteracja} badana zmianna:{wspolczynnik_uczenia},{wspolczynnik_momentum} liczba epok: {liczba_epok}\n')
                        np.set_printoptions(suppress=True)
                        TP, TN, FP, FN, macierz_odpowiedzi = network.validate(test_set, test_labels)
                        f1measure += self.f1measure(TP, TN, FP, FN)
                        accuracy += self.accuracy(TP, TN, FP, FN)
                        precision += self.precision(TP, TN, FP, FN)
                        recall += self.recall(TP, TN, FP, FN)
                        specifity += self.specifity(TP, TN, FP, FN)
                        suma_epok += liczba_epok
                    file.writelines(
                        f'{wspolczynnik_uczenia};{wspolczynnik_momentum};{suma_epok / 10};{accuracy / 10};{precision / 10};{recall / 10};{specifity / 10};{f1measure / 10}\n')

    def test_adagrad(self, train_set, train_labels, test_set, test_labels):
        with open('badania_adagrad_relu.txt', 'w') as file:
            file.writelines(
                'wspolczynnik_uczenia;wspolczynnik momentum;liczba epok;accuracy;precision;recall;specifity;f1measure\n')
            for wspolczynnik_momentum in self.wspolczynnik_momentum:
                for wspolczynnik_uczenia in self.learning_factors:
                    f1measure = 0
                    accuracy = 0
                    precision = 0
                    recall = 0
                    specifity = 0
                    suma_epok = 0
                    print(f'wspolczynnniki:{wspolczynnik_uczenia},{wspolczynnik_momentum}')
                    for iteracja in range(10):
                        network = mlp(self.rozmiary_sieci, [-1, 1], self.biasrange, [4], self.break_rule,
                                      wspolczynnik_uczenia, 3,
                                      wspolczynnik_momentum, self.wspolczynnik_momentum2)
                        liczba_epok = network.uczenie_calosc(train_set, train_labels, epoki=200, batch_size=16)
                        print(
                            f'iteracja: {iteracja} badana zmiana:{wspolczynnik_uczenia},{wspolczynnik_momentum} liczba epok: {liczba_epok}\n')
                        np.set_printoptions(suppress=True)
                        TP, TN, FP, FN, macierz_odpowiedzi = network.validate(test_set, test_labels)
                        f1measure += self.f1measure(TP, TN, FP, FN)
                        accuracy += self.accuracy(TP, TN, FP, FN)
                        precision += self.precision(TP, TN, FP, FN)
                        recall += self.recall(TP, TN, FP, FN)
                        specifity += self.specifity(TP, TN, FP, FN)
                        suma_epok += liczba_epok
                    file.writelines(
                        f'{wspolczynnik_uczenia};{wspolczynnik_momentum};{suma_epok / 10};{accuracy / 10};{precision / 10};{recall / 10};{specifity / 10};{f1measure / 10}\n')

    def test_adadelta(self, train_set, train_labels, test_set, test_labels):
        with open('badania_adadelta_relu.txt', 'w') as file:
            file.writelines(
                'wspolczynnik_uczenia;wspolczynnik momentum;liczba epok;accuracy;precision;recall;specifity;f1measure\n')
            for wspolczynnik_momentum in self.wspolczynnik_momentum:
                for wspolczynnik_uczenia in self.learning_factors:
                    f1measure = 0
                    accuracy = 0
                    precision = 0
                    recall = 0
                    specifity = 0
                    suma_epok = 0
                    print(f'wspolczynnniki:{wspolczynnik_uczenia},{wspolczynnik_momentum}')
                    for iteracja in range(10):
                        network = mlp(self.rozmiary_sieci, [-1, 1], self.biasrange, [4], self.break_rule,
                                      wspolczynnik_uczenia, 4,
                                      wspolczynnik_momentum, self.wspolczynnik_momentum2)
                        liczba_epok = network.uczenie_calosc(train_set, train_labels, epoki=200, batch_size=16)
                        print(
                            f'iteracja: {iteracja} badana zmiana:{wspolczynnik_uczenia},{wspolczynnik_momentum} liczba epok: {liczba_epok}\n')
                        np.set_printoptions(suppress=True)
                        TP, TN, FP, FN, macierz_odpowiedzi = network.validate(test_set, test_labels)
                        f1measure += self.f1measure(TP, TN, FP, FN)
                        accuracy += self.accuracy(TP, TN, FP, FN)
                        precision += self.precision(TP, TN, FP, FN)
                        recall += self.recall(TP, TN, FP, FN)
                        specifity += self.specifity(TP, TN, FP, FN)
                        suma_epok += liczba_epok
                    file.writelines(
                        f'{wspolczynnik_uczenia};{wspolczynnik_momentum};{suma_epok / 10};{accuracy / 10};{precision / 10};{recall / 10};{specifity / 10};{f1measure / 10}\n')

    def test_adam(self, train_set, train_labels, test_set, test_labels):
        with open('badania_adam_sigmoid.txt', 'w') as file:
            file.writelines(
                'wspolczynnik_momentum;wspolczynnik momentum2;liczba epok;accuracy;precision;recall;specifity;f1measure\n')
            for wspolczynnik_momentum in self.wspolczynnik_momentum:
                for wspolczynnik_momentum2 in self.wspolczynnik_momentum2:
                    f1measure = 0
                    accuracy = 0
                    precision = 0
                    recall = 0
                    specifity = 0
                    suma_epok = 0
                    print(f'wspolczynnniki:{wspolczynnik_momentum},{wspolczynnik_momentum2}')
                    for iteracja in range(10):
                        network = mlp(self.rozmiary_sieci, [-1, 1], self.biasrange, [2], self.break_rule, 0.01, 5,
                                      wspolczynnik_momentum, wspolczynnik_momentum2)
                        liczba_epok, _= network.uczenie_calosc(train_set, train_labels, epoki=200, batch_size=16)
                        print(
                            f'iteracja: {iteracja} badana zmiana:{wspolczynnik_momentum},{wspolczynnik_momentum2} liczba epok: {liczba_epok}\n')
                        np.set_printoptions(suppress=True)
                        TP, TN, FP, FN, macierz_odpowiedzi = network.validate(test_set, test_labels)
                        f1measure += self.f1measure(TP, TN, FP, FN)
                        accuracy += self.accuracy(TP, TN, FP, FN)
                        precision += self.precision(TP, TN, FP, FN)
                        recall += self.recall(TP, TN, FP, FN)
                        specifity += self.specifity(TP, TN, FP, FN)
                        suma_epok += liczba_epok
                    file.writelines(
                        f'{wspolczynnik_momentum};{wspolczynnik_momentum2};{suma_epok / 10};{accuracy / 10};{precision / 10};{recall / 10};{specifity / 10};{f1measure / 10}\n')

    def show_momentum_epoki(self, x, y):
        plt.title('Badanie wpływu wspolczynnika momentum dla momentum na uczenie')
        plt.xlabel('wpsolczynnik momentum')
        plt.ylabel('epoki')
        plt.plot(x, y, 'bo', label='średnia liczba epok')
        plt.legend()
        plt.show()

    def show_momentum_procent(self, x, y):
        plt.title('Badanie wpływu wspolczynnika momentum dla momentum na jakość uczenia')
        plt.xlabel('wspolczynnik momentum')
        plt.ylabel('procenty')
        plt.plot(x, y, 'bo', label='średnia liczba procentów')
        plt.legend()
        plt.show()

    def show_momentum_nesterova_epoki(self, x, y):
        plt.title('Badanie wpływu wspolczynnika momentum\n dla momentum Nesterova na uczenie')
        plt.xlabel('wpsolczynnik momentum')
        plt.ylabel('epoki')
        plt.plot(x, y, 'bo', label='średnia liczba epok')
        plt.legend()
        plt.show()

    def show_momentum_nesterova_procent(self, x, y):
        plt.title('Badanie wpływu wspolczynnika momentum\n dla momentum Nesterova na jakość uczenia')
        plt.xlabel('wspolczynnik momentum')
        plt.ylabel('procenty')
        plt.plot(x, y, 'bo', label='średnia liczba procentów')
        plt.legend()
        plt.show()

    def show_adagrad_epoki(self, x, y):
        plt.title('Badanie wpływu wspolczynnika momentum\n dla adagrad na uczenie')
        plt.xlabel('wpsolczynnik momentum')
        plt.ylabel('epoki')
        plt.plot(x, y, 'bo', label='średnia liczba epok')
        plt.legend()
        plt.show()

    def show_adagrad_procent(self, x, y):
        plt.title('Badanie wpływu wspolczynnika momentum\n dla adagrad na jakość uczenia')
        plt.xlabel('wspolczynnik momentum')
        plt.ylabel('procenty')
        plt.plot(x, y, 'bo', label='średnia liczba procentów')
        plt.legend()
        plt.show()
