from layers.conv2d import Conv2D
from layers.relu import ReLU
from layers.sigmoid import Sigmoid


class EdgeNet:

    def __init__(self):

        #####################################
        # Architecture
        #####################################

        self.conv1 = Conv2D(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.relu1 = ReLU()

        self.conv2 = Conv2D(
            in_channels=16,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.sigmoid = Sigmoid()

        #####################################
        # Trainable Layers
        #####################################

        self.layers = [
            self.conv1,
            self.conv2
        ]

    #####################################################

    def forward(self, x):

        x = self.conv1.forward(x)

        x = self.relu1.forward(x)

        x = self.conv2.forward(x)

        x = self.sigmoid.forward(x)

        return x

    #####################################################

    def backward(self, grad):

        grad = self.sigmoid.backward(grad)

        grad = self.conv2.backward(grad)

        grad = self.relu1.backward(grad)

        grad = self.conv1.backward(grad)

        return grad