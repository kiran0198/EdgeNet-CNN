import numpy as xp

from utils.padding import pad2d


def im2col(x, kernel_size, stride=1, padding=0):

    if isinstance(kernel_size, int):
        KH = KW = kernel_size
    else:
        KH, KW = kernel_size

    N, C, H, W = x.shape

    x = pad2d(x, padding)

    H_p = H + 2 * padding
    W_p = W + 2 * padding

    OH = (H_p - KH) // stride + 1
    OW = (W_p - KW) // stride + 1

    cols = xp.zeros((C * KH * KW, N * OH * OW))

    col = 0

    for n in range(N):

        for i in range(0, H_p - KH + 1, stride):

            for j in range(0, W_p - KW + 1, stride):

                patch = x[n, :, i:i+KH, j:j+KW]

                cols[:, col] = patch.reshape(-1)

                col += 1

    return cols