import numpy as xp


class BinaryCrossEntropy:

    def __init__(self, epsilon=1e-12):
        self.epsilon = epsilon

    ########################################################

    def forward(self, y_pred, y_true):

        self.y_pred = xp.clip(
            y_pred,
            self.epsilon,
            1.0 - self.epsilon
        )

        self.y_true = y_true

        loss = -(
            y_true * xp.log(self.y_pred)
            +
            (1 - y_true) * xp.log(1 - self.y_pred)
        )

        return xp.mean(loss)

    ########################################################

    def backward(self):

        N = xp.prod(self.y_true.shape)

        dY = (
            -(self.y_true / self.y_pred)
            +
            ((1 - self.y_true) / (1 - self.y_pred))
        )

        return dY / N