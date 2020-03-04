import jax.numpy as jnp
import numpy as np
import pytest

from pymloc.model.domains import RNDomain
from pymloc.model.dynamical_system.flow_problem import LinearFlow
from pymloc.model.dynamical_system.initial_value_problem import InitialValueProblem
from pymloc.model.dynamical_system.parameter_dae import LinearParameterDAE
from pymloc.model.dynamical_system.parameter_ivp import ParameterInitialValueProblem
from pymloc.model.dynamical_system.representations import LinearFlowRepresentation
from pymloc.model.optimization.constraints.constraint import Constraint
from pymloc.model.optimization.local_optimization import LocalConstraint
from pymloc.model.optimization.local_optimization import LocalNullOptimization
from pymloc.model.optimization.local_optimization import LocalObjective
from pymloc.model.optimization.objectives.objective import Objective
from pymloc.model.optimization.optimization import NullOptimization
from pymloc.model.sensitivities.boundary_dae import BVPSensitivities
from pymloc.model.variables import InputStateVariables
from pymloc.model.variables import NullVariables
from pymloc.model.variables import ParameterContainer
from pymloc.model.variables.container import InputOutputStateVariables
from pymloc.model.variables.container import StateVariablesContainer
from pymloc.model.variables.time_function import StateVariables
from pymloc.model.variables.time_function import Time
from pymloc.solvers.dynamical_systems.sensitivities import SensitivitiesSolver

np.set_printoptions(precision=4)


@pytest.fixture
def bvp_sens_object(linear_param_bvp):
    return BVPSensitivities(linear_param_bvp, n_param=1)


@pytest.fixture
def sens_solver(bvp_sens_object):
    return SensitivitiesSolver(bvp_sens_object, 1e-3)


@pytest.fixture(params=np.arange(0.1, 2., 0.1))
def localized_bvp(bvp_sens_object, request):
    parameters = np.array([request.param])
    return bvp_sens_object.get_sensitivity_bvp(parameters)


@pytest.fixture
def localized_flow_prob(localized_bvp):
    time_interval = Time(0., 1.)
    flowprob = LinearFlow(time_interval, localized_bvp.dynamical_system)
    return flowprob


#param dae
@pytest.fixture
def linear_param_bvp(linear_param_dae, param_vars):
    initial_value = lambda p: jnp.array([p, 2., 1.])
    t = Time(0., 1.)
    return ParameterInitialValueProblem(*param_vars, initial_value, t,
                                        linear_param_dae, 1)


@pytest.fixture
def linear_param_dae(param_vars, e_lin_param_dae, a_lin_param_dae,
                     f_lin_param_dae):
    return LinearParameterDAE(*param_vars, e_lin_param_dae, a_lin_param_dae,
                              f_lin_param_dae, 3)


@pytest.fixture
def param_vars():
    states = StateVariablesContainer(3)
    domain = RNDomain(1)
    parameters = ParameterContainer(1, domain)
    null_vars = NullVariables()
    return null_vars, parameters, states


@pytest.fixture
def initial_value_problem_dae(linear_real_dae):
    initial_value = np.array([1., 2., 0.])
    time_interval = Time(0., 1.)
    return ParameterInitialValueProblem(initial_value, time_interval,
                                        linear_real_dae)


@pytest.fixture
def e_lin_param_dae():
    def e(p, t):
        return np.array([[t, 1., t], [t + 1., 1., 2.],
                         [2 * t + 1., 2., t + 2.]])

    return e


@pytest.fixture
def a_lin_param_dae():
    def a(p, t):
        return -jnp.array([[3., 0., 0.], [0., p, 0.], [0., 0., 10.]])
        #return -np.diag((3., p, 10.))

    return a


@pytest.fixture
def f_lin_param_dae():
    def f(p, t):
        return np.array([t, t**2, -t])

    return f


#real dae rank(e) < n
@pytest.fixture
def linear_real_dae(variables, e_lin_dae, a_lin_dae, f_lin_dae):
    states = StateVariablesContainer(3)
    return LinearFlowRepresentation(states, e_lin_dae, a_lin_dae, f_lin_dae, 3)


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
    return LinearFlowRepresentation(states, e_lin, a_lin, f_lin, 2)


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
    states = StateVariablesContainer(2)
    return LinearFlowRepresentation(states, e_lin, a_lin, f_lin, 2)


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
    variables.states.current_values = variables.states.get_random_values()
    variables.inputs.current_values = variables.inputs.get_random_values()
    variables.outputs.current_values = variables.outputs.get_random_values()
    variables.time.current_values = variables.time.get_random_values()
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
