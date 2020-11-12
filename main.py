from mlp import *
from mnist import MNIST
from Testowanie import *
mndata = MNIST('mnist')
images, labels = mndata.load_training()
# wagi = [784,20,10]
# wagirange = [-1,1]
# biasrange = [-1,1]
# break_rule = 4.5
# epoki =1000
# batch_size = 32
# network = mlp(wagi, wagirange , biasrange, [2], break_rule, 0.1)
# network.display_mlp()
img = images[:50000]
labs = labels[:50000]
valimg = images[59000:60000]
vallabs = labels[59000:60000]
# print('------')
# liczba_epok = network.uczenie_calosc(img, labs, epoki, batch_size)
# print(f'liczba epok: {liczba_epok}')
# np.set_printoptions(suppress=True)
# procent, macierz_odpowiedzi = network.validate(valimg, vallabs)
# print(f'{procent}%')
# print(macierz_odpowiedzi)
testowanie = Testowanie()
testowanie.test_activation_function(img,labs, valimg, vallabs)

# print(mndata.display(images[0]))
# print(labels[0])
# print(network.softMax([1,1,2,1,4]))
