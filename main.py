from Testowanie2 import *

from mlp import *
from CNN import *
from mnist import MNIST
from Testowanie import *
mndata = MNIST('mnist')
images, labels = mndata.load_training()
img = images[:50000]
labs = labels[:50000]
valimg = images[59000:59100]
vallabs = labels[59000:59100]
print('------')
# wagi = [784, 50, 10]
# wagirange = [-1,1]
# biasrange = [-1,1]
# break_rule = 3
# epoki = 100
# batch_size = 20
# network = mlp(wagi, wagirange , biasrange, [2], break_rule, 0.001, 5, 0.99)
# network.display_mlp



cnn = CNN(10,6,1,1,0.02)
# cnn.run(img,labs,3,1,1)
cnn.uczenie(img, labs, 3, 1, 1, 100, 2)
cnn.run(valimg, vallabs, 3, 1, 1)



# liczba_epok = network.uczenie_calosc(img, labs, epoki, batch_size)
# print(f'liczba epok: {liczba_epok}')
# np.set_printoptions(suppress=True)
# procent, macierz_odpowiedzi = network.validate(valimg, vallabs)
# print(f'{procent}%')
# print(macierz_odpowiedzi)
#testowanie = Testowanie2()
# testowanie.test_momentum(img,labs, valimg, vallabs)
#testowanie.test_momentum_nesterova(img,labs, valimg, vallabs)
#testowanie.test_adagrad(img,labs, valimg, vallabs)
#testowanie.test_adadelta(img,labs, valimg, vallabs)
#testowanie.test_adam(img,labs, valimg, vallabs)
#testowanie.testowe_przejscie(img,labs, valimg, vallabs)

# print(mndata.display(images[0]))
# print(labels[0])
# print(network.softMax([1,1,2,1,4]))
