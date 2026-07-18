import numpy as xp
from utils.im2col import im2col
from utils.col2im import col2im


class Conv2D:

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0
    ):

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # he Initialization
        scale = xp.sqrt(2.0 / (in_channels * kernel_size * kernel_size))

        self.W = xp.random.randn(
            out_channels,
            in_channels,
            kernel_size,
            kernel_size
        ) * scale

        self.b = xp.zeros((out_channels, 1))

    ########################################################

    def forward(self, x):

        self.x = x

        N, C, H, W = x.shape

        K = self.kernel_size

        H_out = (H + 2*self.padding - K)//self.stride + 1
        W_out = (W + 2*self.padding - K)//self.stride + 1

        self.x_col = im2col(
            x,
            kernel_size=K,
            stride=self.stride,
            padding=self.padding
        )

        self.W_col = self.W.reshape(
            self.out_channels,
            -1
        )

        out = self.W_col @ self.x_col
        out += self.b

        out = out.reshape(
            self.out_channels,
            N,
            H_out,
            W_out
        )

        out = out.transpose(1, 0, 2, 3)

        return out

    ########################################################

    def backward(self, dY):

        N = dY.shape[0]

        dY_col = dY.transpose(
            1,
            0,
            2,
            3
        ).reshape(
            self.out_channels,
            -1
        )

        ####################################
        # db
        ####################################

        self.db = xp.sum(
            dY_col,
            axis=1,
            keepdims=True
        )

        ####################################
        # dW
        ####################################

        dW_col = dY_col @ self.x_col.T

        self.dW = dW_col.reshape(
            self.W.shape
        )

        ####################################
        # dX
        ####################################

        dX_col = self.W_col.T @ dY_col

        dX = col2im(
            dX_col,
            self.x.shape,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding
        )

        return dX

    ########################################################

    def step(self, lr):

        self.W -= lr * self.dW

        self.b -= lr * self.db