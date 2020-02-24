import numpy as np
import pytest

from pymloc.model.control_system.dae import LinearParameterControlSystem


@pytest.fixture
def e():
    def e(p, x, t):
        return np.identity(2)

    return e


@pytest.fixture
def a():
    def a(p, x, t):
        return -np.diag((3., 1.))

    return a


@pytest.fixture
def b():
    def b(p, x, t):
        return np.identity(2)

    return b


@pytest.fixture
def c():
    def c(p, x, t):
        return np.identity(2)

    return c


@pytest.fixture
def d():
    def d(p, x, t):
        return np.identity(2)

    return d


@pytest.fixture
def f():
    def f(p, t):
        return np.zeros((2, 1))

    return f


class TestDAEControlSystem:
    def test_init(self, variables, e, a, b, c, d, f):
        LinearParameterControlSystem(*variables, e, a, b, c, d, f, 2)
