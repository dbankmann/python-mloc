import numpy as np
import pytest

from pymloc.model.control_system.parameter_dae import LinearParameterControlSystem
from pymloc.model.domains import RNDomain
from pymloc.model.optimization.parameter_optimal_control import ParameterDependentOptimalControl
from pymloc.model.optimization.parameter_optimal_control import ParameterLQRConstraint
from pymloc.model.optimization.parameter_optimal_control import ParameterLQRObjective
from pymloc.model.variables import InputStateVariables
from pymloc.model.variables import NullVariables
from pymloc.model.variables import ParameterContainer
from pymloc.model.variables.time_function import Time


@pytest.fixture
def variables():
    loc_vars = InputStateVariables(1, 1)
    hl_vars = ParameterContainer(1, domain=RNDomain(1))
    return hl_vars, loc_vars


@pytest.fixture
def param_control(variables):

    ll_vars = NullVariables()

    def e(p, t):
        return np.array([[1.]])

    def a(p, t):
        return np.array([[-1.]])

    def b(p, t):
        return np.array([[1.]])

    def c(p, t):
        return np.array([[1.]])

    def d(p, t):
        return np.array([[0.]])

    def f(p, t):
        return np.array([0.])

    return LinearParameterControlSystem(ll_vars, *variables, e, a, b, c, d, f)


@pytest.fixture
def initial_value():
    def ivp(p):
        return np.array([[p]])

    return ivp


@pytest.fixture
def q():
    def q(p, t):
        return np.array([[p, 0.], [0., 1.]])

    return q


@pytest.fixture
def s():
    def s(p, t):
        return np.zeros((2, 1))

    return s


@pytest.fixture
def r():
    def r(p, t):
        return np.array([[1.]])

    return r


@pytest.fixture
def m():
    def m(p):
        return np.array([[0.]])


@pytest.fixture
def pdoc_objective(q, s, r, m, variables):
    time = Time(0., 10.)
    return ParameterLQRObjective(*variables, time, q, s, r, m)


@pytest.fixture
def pdoc_constraint(variables, param_control, initial_value):
    return ParameterLQRConstraint(*variables, param_control, initial_value)


@pytest.fixture
def pdoc_object(variables, pdoc_objective, pdoc_constraint):
    return ParameterDependentOptimalControl(*variables, pdoc_objective,
                                            pdoc_constraint)


class TestPDOCObject:
    def test_eval_weights(self, pdoc_object):
        random_p, = pdoc_object.parameters.get_random_values()
        random_x, random_u, random_t = pdoc_object.state_input.get_random_values(
        )
        local_lq = pdoc_object.get_localized_object(hl_value=random_p)
        weights = local_lq.objective.integral_weights(1.)
        assert weights.shape == (3, 3) and weights[0, 0] == random_p[0]
