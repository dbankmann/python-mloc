import numpy as np
import pytest

from pymloc.model.domains import RNDomain
from pymloc.model.dynamical_system.dae import LinearParameterDAE
from pymloc.model.dynamical_system.flow_problem import LinearFlow
from pymloc.model.dynamical_system.initial_value_problem import \
    InitialValueProblem
from pymloc.model.dynamical_system.representations import \
    LinearFlowRepresenation
from pymloc.model.optimization.constraints.constraint import Constraint
from pymloc.model.optimization.local_optimization import (LocalConstraint,
                                                          LocalNullOptimization,
                                                          LocalObjective)
from pymloc.model.optimization.objectives.objective import Objective
from pymloc.model.optimization.optimization import NullOptimization
from pymloc.model.variables import (InputStateVariables, NullVariables,
                                    ParameterContainer)
from pymloc.model.variables.container import InputOutputStateVariables
from pymloc.model.variables.time_function import StateVariables, Time

np.set_printoptions(precision=4)


@pytest.fixture
def linear_real_dae(variables, e_lin_dae, a_lin_dae, f_lin_dae):
    states = StateVariables(3)
    return LinearFlowRepresenation(states, e_lin_dae, a_lin_dae, f_lin_dae, 3)


@pytest.fixture
def flow_problem_dae(linear_real_dae):
    time_interval = Time(0., 1.)
    flowprob = LinearFlow(time_interval, linear_real_dae)
    return flowprob


@pytest.fixture
def initial_value_problem_dae(linear_real_dae):
    initial_value = np.array([1., 2., 0.])
    time_interval = Time(0., 1.)
    return InitialValueProblem(initial_value, time_interval, linear_real_dae)


@pytest.fixture
def e_lin_dae():
    def e(t):
        return np.array([[t, 1., t], [t + 1., 1., 2.],
                         [2 * t + 1., 2., t + 2.]])

    return e


@pytest.fixture
def a_lin_dae():
    def a(t):
        return -np.diag((3., 1., 10.))

    return a


@pytest.fixture
def f_lin_dae():
    def f(t):
        return np.array([t, t**2, -t])

    return f


#ODE
@pytest.fixture
def linear_dae(variables, e_lin, a_lin, f_lin):
    states = StateVariables(2)
    return LinearFlowRepresenation(states, e_lin, a_lin, f_lin, 2)


@pytest.fixture
def flow_problem(linear_dae):
    time_interval = Time(0., 1.)
    flowprob = LinearFlow(time_interval, linear_dae)
    return flowprob


@pytest.fixture
def initial_value_problem(linear_dae):
    initial_value = np.array([1., 2.])
    time_interval = Time(0., 1.)
    return InitialValueProblem(initial_value, time_interval, linear_dae)


@pytest.fixture
def e_lin():
    def e(t):
        return np.identity(2)

    return e


@pytest.fixture
def a_lin():
    def a(t):
        return -np.diag((3., 1.))

    return a


@pytest.fixture
def f_lin():
    def f(t):
        return np.zeros((2, ))

    return f


@pytest.fixture
def linear_dae(variables, e_lin, a_lin, f_lin):
    states = StateVariables(2)
    return LinearFlowRepresenation(states, e_lin, a_lin, f_lin, 2)


@pytest.fixture
def variables():
    domain = RNDomain(1)
    parameters = ParameterContainer(1, domain)
    state_input = InputStateVariables(n_states=2, m_inputs=1)
    null_vars = NullVariables()
    return null_vars, parameters, state_input


@pytest.fixture
def local_opt():
    variables = InputOutputStateVariables(2, 4, 5)
    constraint = LocalConstraint()
    objective = LocalObjective()
    return LocalNullOptimization(objective, constraint,
                                 variables), variables, constraint, objective


@pytest.fixture
def opt():
    variables = InputOutputStateVariables(2, 4, 5)
    constraint = Constraint(*3 * (variables, ))
    objective = Objective(*3 * (variables, ))
    return NullOptimization(objective, constraint, variables, variables,
                            variables), variables, constraint, objective


def pytest_runtest_makereport(item, call):
    if "incremental" in item.keywords:
        if call.excinfo is not None:
            parent = item.parent
            parent._previousfailed = item


def pytest_runtest_setup(item):
    if "incremental" in item.keywords:
        previousfailed = getattr(item.parent, "_previousfailed", None)
        if previousfailed is not None:
            pytest.xfail("previous test failed ({})".format(
                previousfailed.name))
