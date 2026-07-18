import numpy as xp


class Sigmoid:

    def __init__(self):
        self.out = None

    def forward(self, x):
        """
        Forward Sigmoid
        """

        self.out = 1.0 / (1.0 + xp.exp(-x))

        return self.out

    def backward(self, dY):
        """
        Backward Sigmoid
        """

        return dY * self.out * (1.0 - self.out)