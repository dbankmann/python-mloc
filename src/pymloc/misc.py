#
# Copyright (c) 2019-2020
#
# @author: Daniel Bankmann
# @company: Technische Universität Berlin
#
# This file is part of the python package pymloc
# (see https://gitlab.tubit.tu-berlin.de/bankmann91/python-mloc )
#
# License: 3-clause BSD, see https://opensource.org/licenses/BSD-3-Clause
#
import numpy as np


def unstack(array):
    return np.moveaxis(array, -1, 0).reshape(-1, *array.shape[1:-1])


def restack(array, shape):
    *shapefirst, shapem2, shapem1 = shape
    restacked = np.einsum('ij...->j...i',
                          array.reshape(shapem1, *shapefirst, shapem2))
    return restacked
