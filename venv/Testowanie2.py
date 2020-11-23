import matplotlib.pyplot as plt
from mlp import *

class Testowanie2:
    def __init__(self):
        self.hidden_layer_sizes = [1,2,3,5,7,10,12,14,16,18,20,30,40,50,70,90,
                          100,200,300,400,500,600,700,800]
        self.batch_sizes = [1,2,4,8,10,16,20,30,32,40,50,60,64,70,80,90,100,128,200,256,300,400,512]
        self.weights_ranges = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,5,10]
        self.learning_factors = [0.0001,0.0005,0.001,0.005,0.01,0.02,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.5,2]
        self.activation_functions =[1,2,3,4]
        self.break_rule = 0.4
        self.biasrange = [-1,1]
        self.x_epoki = []
        self.x_procenty = []
        self.rozmiary_sieci = [784,70,10]
        self.wspolczynnik_momentum = [0.9]
        self.wspolczynnik_momentum2 = [0.99]
        self.opcja_momentum = [1,2,3,4,5]

    def test_momentum(self, train_set, train_labels, test_set, test_labels):
        with open('badania_momentum.txt', 'w') as file:
            file.writelines('wspolczynnik momentum;liczba epok;poprawne_odpowiedzi\n')
            for wspolczynnik_momentum in self.wspolczynnik_momentum:
                suma_procentów = 0
                suma_epok = 0
                print(f'wspolczynnnik momentum: {wspolczynnik_momentum}')
                for iteracja in range(10):

                    network = mlp([784, 40, 10], [-1, 1], self.biasrange, [2], self.break_rule, 0.01,1, wspolczynnik_momentum, self.wspolczynnik_momentum2)
                    liczba_epok = network.uczenie_calosc(train_set, train_labels, epoki=200, batch_size=32)
                    print(f'iteracja: {iteracja} badana zmianna:{wspolczynnik_momentum} liczba epok: {liczba_epok}\n')
                    np.set_printoptions(suppress=True)
                    procent, macierz_odpowiedzi = network.validate(test_set, test_labels)
                    suma_epok += liczba_epok
                    suma_procentów += procent
                self.x_epoki.append(suma_epok / 10)
                self.x_procenty.append(suma_procentów / 10)
                file.writelines(f'{wspolczynnik_momentum};{suma_epok / 10};{suma_procentów / 10}\n')
            self.show_momentum_procent(self.wspolczynnik_momentum, self.x_procenty)
            self.show_momentum_epoki(self.wspolczynnik_momentum, self.x_epoki)

    def test_momentum_nesterova(self, train_set, train_labels, test_set, test_labels):
        with open('badania_momentum_nesterova.txt', 'w') as file:
            file.writelines('wspolczynnik momentum;liczba epok;poprawne_odpowiedzi\n')
            for wspolczynnik_momentum in self.wspolczynnik_momentum:
                suma_procentów = 0
                suma_epok = 0
                print(f'wspolczynnnik momentum: {wspolczynnik_momentum}')
                for iteracja in range(10):
                    network = mlp([784, 50, 10], [-1, 1], self.biasrange, [2], self.break_rule, 0.01,2, wspolczynnik_momentum, self.wspolczynnik_momentum2)
                    liczba_epok = network.uczenie_calosc(train_set, train_labels, epoki=200, batch_size=32)
                    print(f'iteracja: {iteracja} badana zmianna:{wspolczynnik_momentum} liczba epok: {liczba_epok}\n')
                    np.set_printoptions(suppress=True)
                    procent, macierz_odpowiedzi = network.validate(test_set, test_labels)
                    suma_epok += liczba_epok
                    suma_procentów += procent
                self.x_epoki.append(suma_epok / 10)
                self.x_procenty.append(suma_procentów / 10)
                file.writelines(f'{wspolczynnik_momentum};{suma_epok / 10};{suma_procentów / 10}\n')
            self.show_momentum_nesterova_procent(self.wspolczynnik_momentum, self.x_procenty)
            self.show_momentum_nesterova_epoki(self.wspolczynnik_momentum, self.x_epoki)

    def test_adagrad(self, train_set, train_labels, test_set, test_labels):
        with open('badania_adagrad.txt', 'w') as file:
            file.writelines('wspolczynnik momentum;liczba epok;poprawne_odpowiedzi\n')
            for wspolczynnik_momentum in self.wspolczynnik_momentum:
                suma_procentów = 0
                suma_epok = 0
                print(f'wspolczynnnik momentum: {wspolczynnik_momentum}')
                for iteracja in range(10):

                    network = mlp([784, 50, 10], [-1, 1], self.biasrange, [2], self.break_rule, 0.01,3, wspolczynnik_momentum, self.wspolczynnik_momentum2)
                    liczba_epok = network.uczenie_calosc(train_set, train_labels, epoki=50, batch_size=16)
                    print(f'iteracja: {iteracja} badana zmianna:{wspolczynnik_momentum} liczba epok: {liczba_epok}\n')
                    np.set_printoptions(suppress=True)
                    procent, macierz_odpowiedzi = network.validate(test_set, test_labels)
                    suma_epok += liczba_epok
                    suma_procentów += procent
                self.x_epoki.append(suma_epok / 10)
                self.x_procenty.append(suma_procentów / 10)
                file.writelines(f'{wspolczynnik_momentum};{suma_epok / 10};{suma_procentów / 10}\n')
            self.show_adagrad_procent(self.wspolczynnik_momentum, self.x_procenty)
            self.show_adagrad_epoki(self.wspolczynnik_momentum, self.x_epoki)
    def show_momentum_epoki(self,x, y):
        plt.title('Badanie wpływu wspolczynnika momentum dla momentum na uczenie')
        plt.xlabel('wpsolczynnik momentum')
        plt.ylabel('epoki')
        plt.plot(x, y, 'bo', label='średnia liczba epok')
        plt.legend()
        plt.show()

    def show_momentum_procent(self,x, y):
        plt.title('Badanie wpływu wspolczynnika momentum dla momentum na jakość uczenia')
        plt.xlabel('wspolczynnik momentum')
        plt.ylabel('procenty')
        plt.plot(x, y, 'bo', label='średnia liczba procentów')
        plt.legend()
        plt.show()

    def show_momentum_nesterova_epoki(self,x, y):
        plt.title('Badanie wpływu wspolczynnika momentum\n dla momentum Nesterova na uczenie')
        plt.xlabel('wpsolczynnik momentum')
        plt.ylabel('epoki')
        plt.plot(x, y, 'bo', label='średnia liczba epok')
        plt.legend()
        plt.show()

    def show_momentum_nesterova_procent(self,x, y):
        plt.title('Badanie wpływu wspolczynnika momentum\n dla momentum Nesterova na jakość uczenia')
        plt.xlabel('wspolczynnik momentum')
        plt.ylabel('procenty')
        plt.plot(x, y, 'bo', label='średnia liczba procentów')
        plt.legend()
        plt.show()

    def show_adagrad_epoki(self,x, y):
        plt.title('Badanie wpływu wspolczynnika momentum\n dla adagrad na uczenie')
        plt.xlabel('wpsolczynnik momentum')
        plt.ylabel('epoki')
        plt.plot(x, y, 'bo', label='średnia liczba epok')
        plt.legend()
        plt.show()

    def show_adagrad_procent(self,x, y):
        plt.title('Badanie wpływu wspolczynnika momentum\n dla adagrad na jakość uczenia')
        plt.xlabel('wspolczynnik momentum')
        plt.ylabel('procenty')
        plt.plot(x, y, 'bo', label='średnia liczba procentów')
        plt.legend()
        plt.show()