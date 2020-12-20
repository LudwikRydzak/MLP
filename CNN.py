from mlp import *
import numpy as np

class CNN:
    def __init__(self, filter_number, filter_size, step, frame_width, learning_factor):
        self.network = None
        self.filter_number = filter_number
        self.filter_size = filter_size
        self.step = step
        self.frame_width = frame_width
        self.learning_factor = learning_factor
        self.filters = []
        for i in range(self.filter_number):
            self.filters.append(self.inicjacja_wag(2,3,3))

    def get_random_from_range(self, start_end):
        return (random.random() * (start_end[1] - start_end[0]) + start_end[0])

    def inicjacja_wag(self, ini, input_size, output_size):
        if (ini == 1):  # xavier
            var = 2 / (input_size + output_size)
        elif (ini == 2):  # he
            var = 2 / input_size
        else:
            var = 1
        return np.random.normal(0, var, size=(input_size, output_size))

    def convolute(self, filter_size,filters, image):
        image  = np.reshape(image,(28,28))
        map_list = []
        #ustalam na sztywno frame, step i szerokosc obrazka bo nie zmieniam ich w tym cwiczeniu
        step =1
        frame_width =1
        map_size = int(((28-filter_size+2)/1)+1)
        for i in range(len(filters)):
            feature_map = np.zeros((map_size, map_size))
            map_list.append(feature_map)

        for map_number in range(len(map_list)):
            for x in range(map_size):
                for y in range(map_size):
                    pobudzenie = 0
                    for a in  range(filter_size):
                        for b in range(filter_size):
                            if((x+a)<len(image) and (y+b)< len(image)):
                                pobudzenie += image[x+a][y+b] * (filters[map_number])[a][b]
                    # (map_list[map_number])[x][y] = self.relu(pobudzenie)
                    (map_list[map_number])[x][y] = pobudzenie
        return map_list

    def relu(self, pobudzenie):
        if(pobudzenie >0):
            return pobudzenie
        else:
            return 0

    def max_pooling(self, map_list):
        max_values_list = []
        max_indexes_x_list= []
        max_indexes_y_list= []
        shape = np.shape(map_list[0])
        shape =((int) (shape[0]/2), (int) (shape[1]/2))
        max_values = np.zeros(shape)
        max_indexes_x = np.zeros(shape)
        max_indexes_y = np.zeros(shape)
        for single_map in map_list:
            for i in range(shape[0]):
                for j in range(shape[1]):
                    maximum = max(single_map[i * 2][j * 2], single_map[i * 2 + 1][j * 2], single_map[i * 2][j * 2 + 1],
                                  single_map[i * 2 + 1][j * 2 + 1])
                    if (maximum == single_map[i * 2][j * 2]):
                        max_indexes_x[i][j]= i * 2
                        max_indexes_y[i][j]= j * 2
                    elif (maximum == single_map[i * 2 + 1][j * 2]):
                        max_indexes_x[i][j]= i * 2 + 1
                        max_indexes_y[i][j]= j * 2
                    elif (maximum == single_map[i * 2][j * 2 + 1]):
                        max_indexes_x[i][j]= i * 2
                        max_indexes_y[i][j]= j * 2 + 1
                    elif (maximum == single_map[i * 2 + 1][j * 2 + 1]):
                        max_indexes_x[i][j]= i * 2 + 1
                        max_indexes_y[i][j]= j * 2 + 1
                    max_values[i][j] = maximum
            max_values_list.append(max_values)
            max_indexes_x_list.append(max_indexes_x)
            max_indexes_y_list.append(max_indexes_y)
        return max_values_list, max_indexes_x_list, max_indexes_y_list

    def run(self, images, labels, filter_size, step, frame_width):
        suma = 0
        for indeks in range(len(images)):
            map_list = self.convolute(filter_size, self.filters, images[indeks])
            max_values_list, max_indexes_x_list, max_indexes_y_list = self.max_pooling(map_list)
            input = []
            map_size = int((((28-filter_size+2)/1)+1)/2)
            input_size = map_size * map_size * self.filter_number

            for i in range(len(max_values_list)):
                input.append(np.reshape(max_values_list[i],-1))
            input = np.reshape(input, -1)
            wynik = self.network.wynikowa_etykieta(self.network.forward_propagation(input))
            if(wynik == labels[indeks]):
                suma+=1
        print(suma)

    def uczenie(self, images, labels, filter_size, step, frame_width, epoki, batch_size):
        table = [0] * 200
        inputs = []
        map_size = int((((28-filter_size+2)/1)+1)/2)
        input_size = map_size * map_size * self.filter_number
        self.network = mlp([input_size, input_size, 20, 10], [-1, 1], [-1, 1], [2,2,2], 1,
                                  0.01, 1, 0, 0.999,3)
        for epoka in range(epoki):
            randomstart = round(self.get_random_from_range([0, 50000 - batch_size - 1]))
            minibatch_etykiety = labels[randomstart:randomstart + batch_size]
            minibatch_wejscia = images[randomstart:randomstart + batch_size]
            for indeks in range(batch_size):
                map_list = self.convolute(filter_size, self.filters, minibatch_wejscia[indeks])
                max_values_list, max_indexes_x_list, max_indexes_y_list = self.max_pooling(map_list)
                input = []
                for i in range(len(max_values_list)):
                    input.append(np.reshape(max_values_list[i],-1))
                input = np.reshape(input, input_size)
                inputs.append(input)
            wejscia = np.asarray(inputs)
            loss = 0
            for i in range(len(wejscia)):
                loss += self.network.uczenie(wejscia[i], labels[i])
                self.convolution_change(images[i],self.upsample(self.network.errors[0],max_indexes_x_list, max_indexes_y_list))
            self.network.zamiana_wag()
            self.network.poprzednia_zmiana_wag = np.copy(self.network.zmiana_wag)
            inputs = []
            print(f"epoka: {epoka} error: {( loss / len(wejscia))}")


    def upsample(self, error_vector,max_indexesx, max_indexesy):
        temp_length = len(max_indexesx[0])
        upsampleds = []

        upsampled = np.reshape(np.zeros((temp_length*2)**2),(temp_length*2,temp_length*2))
        error_vector = np.reshape(error_vector,(self.filter_number,temp_length,temp_length) )
        for number in range(len(max_indexesx)):
            for i in range(len(max_indexesx[0])):
                for j in range(len(max_indexesx[0][0])):

                    upsampled[(int)(max_indexesx[number][i][j])][(int)(max_indexesy[number][i][j])]+=error_vector[number][i][j]

            upsampleds.append(upsampled)

        return upsampleds

    def convolution_change(self, image, upsampled_error_list):
        rotated_upsampled_list = []
        gradient_list = []
        for i in range(len(upsampled_error_list)):
            rotated_upsampled_list.append(np.rot90(upsampled_error_list,2))
        gradient_list = ( self.convolute(np.shape(upsampled_error_list[0])[0], upsampled_error_list, image))
        for i in range(len(upsampled_error_list)):
            self.filters[i] -= gradient_list[i]*self.learning_factor



