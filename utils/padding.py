import numpy as xp


def pad2d(x, padding):

    if padding == 0:
        return x

    return xp.pad(
        x,
        pad_width=((0, 0), (0, 0), (padding, padding), (padding, padding)),
        mode="constant",
        constant_values=0,
    )


def unpad2d(x, padding):

    if padding == 0:
        return x

    return x[:, :, padding:-padding, padding:-padding]