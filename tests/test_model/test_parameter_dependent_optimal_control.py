import pytest

from pymloc.model.optimization.parameter_optimal_control import ParameterDependentOptimalControl, PDOCConstraint, PDOCObjective

import numpy as np

@pytest.fixture
def q():
    def q(p, x, t):
        return np.array([[p[0], 0.], [0., 1.]])

    return q


@pytest.fixture
def s():
    def s(p, x, t):
        return np.zeros((2, 1))

    return s


@pytest.fixture
def r():
    def r(p, x, t):
        return np.array([[1.]])

    return r



@pytest.fixture
def pdoc_objective(q, s, r, variables):
    return PDOCObjective(*variables, q, s, r)


@pytest.fixture
def pdoc_constraint(variables):
    return PDOCConstraint(*variables)


@pytest.fixture
def pdoc_object(variables, pdoc_objective, pdoc_constraint):
    return ParameterDependentOptimalControl(*(variables[1:]), pdoc_objective,
                                            pdoc_constraint)


class TestPDOCObject:
    def test_eval_weights(self, pdoc_object):
        random_p, = pdoc_object.parameters.get_random_values()
        random_x, random_u, random_t = pdoc_object.state_input.get_random_values(
        )
        weights = pdoc_object.objective._eval_weights(random_p, random_x,
                                                      random_t)
        assert weights.shape == (3, 3)
