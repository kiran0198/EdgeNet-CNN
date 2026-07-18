import numpy as xp

from utils.padding import unpad2d


def col2im(cols, input_shape, kernel_size, stride=1, padding=0):
    """
    Convert column matrix back to image.

    Parameters
    ----------
    cols : np.ndarray
        Shape:
        (C * KH * KW,
         N * OH * OW)

    input_shape : tuple
        (N, C, H, W)

    kernel_size : int or tuple

    stride : int

    padding : int

    Returns
    -------
    np.ndarray
        Shape (N, C, H, W)
    """

    if isinstance(kernel_size, int):
        KH = KW = kernel_size
    else:
        KH, KW = kernel_size

    N, C, H, W = input_shape

    H_pad = H + 2 * padding
    W_pad = W + 2 * padding

    OH = (H_pad - KH) // stride + 1
    OW = (W_pad - KW) // stride + 1

    x = xp.zeros((N, C, H_pad, W_pad))

    col = 0

    for n in range(N):

        for i in range(0, H_pad - KH + 1, stride):

            for j in range(0, W_pad - KW + 1, stride):

                patch = cols[:, col].reshape(C, KH, KW)

                x[n, :, i:i+KH, j:j+KW] += patch

                col += 1

    return unpad2d(x, padding)