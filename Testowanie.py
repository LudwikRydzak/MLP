import matplotlib.pyplot as plt
from mlp import *
class Testowanie:
    def __init__(self):
        self.hidden_layer_sizes = [1,2,3,5,7,10,12,14,16,18,20,30,40,50,70,90,
                          100,200,300,400,500,600,700,800]
        self.batch_sizes = [1,2,4,8,10,16,20,30,32,40,50,60,64,70,80,90,100,128,200,256,300,400,512]
        self.weights_ranges = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,5,10]
        self.learning_factors = [0.0001,0.0005,0.001,0.005,0.01,0.02,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.5,2]
        self.activation_functions =[1,2,3,4]
        self.break_rule = 5
        self.biasrange = [-1,1]
        self.x_epoki = []
        self.x_procenty = []
        self.rozmiary_sieci = [784,70,10]
        self.wspolczynnik_momentum = 0.8

    def test_hidden_layer(self, train_set, train_labels, test_set, test_labels):
        with open('badania_momentum.txt', 'w') as file:
            file.writelines('liczba neuronów warstwy ukrytej;liczba epok;poprawne_odpowiedzi\n')
            for size in self.hidden_layer_sizes:
                suma_procentów =0
                suma_epok =0
                print(f'rozmiar ukrytej: {size}')
                for iteracja in range(10):
                    #         rozmiar macierzy  wagi_range         sigmoid        współczynnik uczenia
                    network = mlp([784,size,10], [-1,1], self.biasrange, [2], self.break_rule, 0.1, )
                    liczba_epok = network.uczenie_calosc(train_set, train_labels, epoki=500, batch_size=32)
                    print(f'iteracja: {iteracja} badana zmianna:{size} liczba epok: {liczba_epok}\n')
                    np.set_printoptions(suppress=True)
                    procent, macierz_odpowiedzi = network.validate(test_set, test_labels)
                    suma_epok += liczba_epok
                    suma_procentów += procent
                self.x_epoki.append(suma_epok/10)
                self.x_procenty.append(suma_procentów/10)
                file.writelines(f'{size};{suma_epok/10};{suma_procentów/10}\n')
            self.show_hidden_size_procent(self.hidden_layer_sizes,self.x_procenty)
            self.show_hidden_size_epoki(self.hidden_layer_sizes,self.x_epoki)

    def show_hidden_size_procent(self,x, y):
        plt.title('Badanie wpływu rozmiaru warstwy ukrytej na jakość uczenia')
        plt.xlabel('rozmiar warstwy ukrytej')
        plt.ylabel('poprawne odpowiedzi')
        plt.plot(x, y, 'bo', label='średnia jakość w procentach')
        plt.legend()
        plt.show()

    def show_hidden_size_epoki(self,x, y):
        plt.title('Badanie wpływu rozmiaru warstwy ukrytej na szybkość uczenia')
        plt.xlabel('rozmiar warstwy ukrytej')
        plt.ylabel('epoki')
        plt.plot(x, y, 'bo', label='średnia liczba epok')
        plt.legend()
        plt.show()

    def test_batch_size(self, train_set, train_labels, test_set, test_labels):
        with open('badania_batch_size.txt', 'w') as file:
            file.writelines('rozmiar batcha;liczba epok;poprawne_odpowiedzi\n')
            for batch_size in self.batch_sizes:
                suma_procentow =0
                suma_epok =0
                for iteracja in range(10):
                    #                                  wagi_range           sigmoid        współczynnik uczenia
                    network = mlp(self.rozmiary_sieci, [-1,1], self.biasrange, [2], self.break_rule, 0.1)
                    liczba_epok = network.uczenie_calosc(train_set, train_labels, 500, batch_size)
                    print(f'iteracja: {iteracja} badana zmianna:{batch_size} liczba epok: {liczba_epok}\n')
                    procent, macierz_odpowiedzi = network.validate(test_set, test_labels)
                    suma_epok += liczba_epok
                    suma_procentow += procent
                self.x_epoki.append(suma_epok/10)
                self.x_procenty.append(suma_procentow/10)
                file.writelines(f'{batch_size};{suma_epok/10};{suma_procentow/10}')
            self.show_batch_size_procent(self.batch_sizes,self.x_procenty)
            self.show_batch_size_epoki(self.batch_sizes,self.x_epoki)

    def show_batch_size_procent(self,x, y):
        plt.title('Badanie wpływu rozmiaru batcha na jakość uczenia')
        plt.xlabel('rozmiar batcha')
        plt.ylabel('poprawne odpowiedzi')
        plt.plot(x, y, 'bo', label='średnia jakość w procentach')
        plt.legend()
        plt.show()

    def show_batch_size_epoki(self,x, y):
        plt.title('Badanie wpływu rozmiaru batcha na szybkość uczenia')
        plt.xlabel('rozmiar batcha')
        plt.ylabel('epoki')
        plt.plot(x, y, 'bo', label='średnia liczba epok')
        plt.legend()
        plt.show()

    def test_weights_range(self, train_set, train_labels, test_set, test_labels):
        with open('badania_weights_range.txt', 'w') as file:
            file.writelines('zakres wag;liczba epok;poprawne_odpowiedzi')
            for ranges in self.weights_ranges:
                suma_procentow =0
                suma_epok =0
                for iteracja in range(10):
                    #                                  wagi_range           sigmoid        współczynnik uczenia
                    network = mlp(self.rozmiary_sieci, [-ranges,ranges], self.biasrange, [2], self.break_rule, 0.1)
                    liczba_epok = network.uczenie_calosc(train_set, train_labels, 500, 20)
                    print(f'iteracja: {iteracja} badana zmianna:{ranges} liczba epok: {liczba_epok}\n')
                    procent, macierz_odpowiedzi = network.validate(test_set, test_labels)
                    suma_epok += liczba_epok
                    suma_procentow += procent
                self.x_epoki.append(suma_epok/10)
                self.x_procenty.append(suma_procentow/10)
                file.writelines(f'{ranges};{suma_epok/10};{suma_procentow/10}')
            self.show_weights_range_procent(self.weights_ranges,self.x_procenty)
            self.show_weights_range_epoki(self.weights_ranges,self.x_epoki)

    def show_weights_range_procent(self,x, y):
        plt.title('Badanie wpływu zakresu wag na jakość uczenia')
        plt.xlabel('zakres wag')
        plt.ylabel('poprawne odpowiedzi')
        plt.plot(x, y, 'bo', label='średnia jakość w procentach')
        plt.legend()
        plt.show()

    def show_weights_range_epoki(self,x, y):
        plt.title('Badanie wpływu zakresu wag na szybkość uczenia')
        plt.xlabel('zakres wag')
        plt.ylabel('epoki')
        plt.plot(x, y, 'bo', label='średnia liczba epok')
        plt.legend()
        plt.show()

    def test_learning_factor(self, train_set, train_labels, test_set, test_labels):
        with open('badania_learning_factor.txt', 'w') as file:
            file.writelines('współczynnik uczenia;liczba epok;poprawne_odpowiedzi')
            for learning_factor in self.learning_factors:
                suma_procentow =0
                suma_epok =0
                for iteracja in range(10):
                    #                                  wagi_range           sigmoid        współczynnik uczenia
                    network = mlp(self.rozmiary_sieci, [-1,1], self.biasrange, [2], self.break_rule, learning_factor)
                    liczba_epok = network.uczenie_calosc(train_set, train_labels, 500, 20)
                    print(f'iteracja: {iteracja} badana zmianna:{learning_factor} liczba epok: {liczba_epok}\n')
                    procent, macierz_odpowiedzi = network.validate(test_set, test_labels)
                    suma_epok += liczba_epok
                    suma_procentow += procent
                self.x_epoki.append(suma_epok/10)
                self.x_procenty.append(suma_procentow/10)
                file.writelines(f'{learning_factor};{suma_epok/10};{suma_procentow/10}')
            self.show_learning_factor_procent(self.learning_factors,self.x_procenty)
            self.show_learning_factor_epoki(self.learning_factors,self.x_epoki)

    def show_learning_factor_procent(self,x, y):
        plt.title('Badanie wpływu współczynnika uczenia na jakość uczenia')
        plt.xlabel('współczynnik uczenia')
        plt.ylabel('poprawne odpowiedzi')
        plt.plot(x, y, 'bo', label='średnia jakość w procentach')
        plt.legend()
        plt.show()

    def show_learning_factor_epoki(self,x, y):
        plt.title('Badanie wpływu współczynnika uczenia na szybkość uczenia')
        plt.xlabel('współczynnik uczenia')
        plt.ylabel('epoki')
        plt.plot(x, y, 'bo', label='średnia liczba epok')
        plt.legend()
        plt.show()

    def test_activation_function(self, train_set, train_labels, test_set, test_labels):
        with open('badania_activation_function.txt', 'w') as file:
            file.writelines('funkcja aktywacji;liczba epok;poprawne_odpowiedzi')
            for activation_function in self.activation_functions:
                suma_procentow =0
                suma_epok =0
                for iteracja in range(10):
                    #                                  wagi_range           sigmoid        współczynnik uczenia
                    network = mlp(self.rozmiary_sieci, [-1,1], self.biasrange, [activation_function], self.break_rule, 0.01)
                    liczba_epok = network.uczenie_calosc(train_set, train_labels, 500, 20)
                    print(f'iteracja: {iteracja} badana zmianna:{activation_function} liczba epok: {liczba_epok}\n')
                    procent, macierz_odpowiedzi = network.validate(test_set, test_labels)
                    suma_epok += liczba_epok
                    suma_procentow += procent
                self.x_epoki.append(suma_epok/10)
                self.x_procenty.append(suma_procentow/10)
                file.writelines(f'\n{activation_function};{suma_epok/10};{suma_procentow/10}')