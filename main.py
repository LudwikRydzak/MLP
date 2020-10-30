from mlp import *
from mnist import MNIST
mndata = MNIST('mnist')
images, labels = mndata.load_training()
wagi = [5,6,4,6]
wagirange = [-1,1]
biasrange = [-1,1]
break_rule = 0
network = mlp(wagi, wagirange , biasrange, [1,2,2], break_rule)
network.display_mlp()
print('------')
print(network.forward_propagation([5,4,3,2,1]))

print(mndata.display(images[0]))
print(labels[0])
