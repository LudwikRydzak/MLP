from mlp import *
from mnist import MNIST
mndata = MNIST('mnist')
images, labels = mndata.load_training()
wagi = [784,16,10]
wagirange = [-1,1]
biasrange = [-1,1]
break_rule = 0
epoki =1000
batch_size = 50
network = mlp(wagi, wagirange , biasrange, [2], break_rule, 0.1)
# network.display_mlp()
img = images[:50000]
labs = labels[:50000]
valimg = images[50000:60000]
vallabs = labels[50000:60000]
print('------')
network.uczenie_calosc(img, labs, epoki, batch_size)
network.display_mlp()
print(network.validate(valimg, vallabs))
network.display_biases()
# print(mndata.display(images[0]))
# print(labels[0])
# print(network.softMax([1,1,2,1,4]))
