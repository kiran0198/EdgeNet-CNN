import numpy as xp


class ReLU:

    def __init__(self):
        self.mask = None

    def forward(self, x):

        self.mask = x > 0

        return xp.maximum(0, x)

    def backward(self, dY):

        dX = dY.copy()

        dX[~self.mask] = 0

        return dX