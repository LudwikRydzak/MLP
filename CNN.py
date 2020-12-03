from mlp import *


class CNN:
    def __init__(self, filter_number, filter_size, step, frame_width, learning_factor):
        self.filter_number = filter_number
        self.filter_size = filter_size
        self.step = step
        self.frame_width = frame_width
        self.learning_factor = learning_factor

    def convolute(self):

