class SGD:
    """
    Stochastic Gradient Descent Optimizer
    """

    def __init__(self, layers, lr=0.001):
        self.layers = layers
        self.lr = lr

    ########################################################

    def step(self):
        """
        Update all trainable parameters.
        """

        for layer in self.layers:

            if hasattr(layer, "W"):

                layer.W -= self.lr * layer.dW

            if hasattr(layer, "b"):

                layer.b -= self.lr * layer.db

    ########################################################

    def zero_grad(self):
        """
        Clear gradients after every update.
        """

        for layer in self.layers:

            if hasattr(layer, "dW"):
                layer.dW.fill(0)

            if hasattr(layer, "db"):
                layer.db.fill(0)