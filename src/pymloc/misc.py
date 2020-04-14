import numpy as np


def unstack(array):
    return np.moveaxis(array, -1, 0).reshape(-1, *array.shape[1:-1])


def restack(array, shape):
    *shapefirst, shapem2, shapem1 = shape
    restacked = np.einsum('ij...->j...i',
                          array.reshape(shapem1, *shapefirst, shapem2))
    return restacked
