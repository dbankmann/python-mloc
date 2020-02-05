import pytest

from pymloc.model.dynamical_system.dae import LinearDAE, LinearParameterDAE

import numpy as np


@pytest.fixture
def e_lin():
    def e(x, t):
        return np.identity(2)

    return e


@pytest.fixture
def a_lin():
    def a(x, t):
        return -np.diag((3., 1.))

    return a


@pytest.fixture
def f_lin():
    def f(t):
        return np.zeros((2, 1))

    return f


class TestLinearDAE:
    def test_init(self, variables, e_lin, a_lin, f_lin):
        state_input = variables[2]
        LinearDAE(state_input, e_lin, a_lin, f_lin)
