import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pymloc.model.control_system.parameter_dae import LinearParameterControlSystem
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
from pymloc.model.optimization.nonlinear_leastsquares import NonLinearLeastSquares
from pymloc.model.optimization.objectives import NonLinearLeastSquares as NLLSObjective
from pymloc.model.optimization.objectives.objective import Objective
from pymloc.model.optimization.optimization import NullOptimization
from pymloc.model.optimization.parameter_optimal_control import ParameterDependentOptimalControl
from pymloc.model.optimization.parameter_optimal_control import ParameterLQRConstraint
from pymloc.model.optimization.parameter_optimal_control import ParameterLQRObjective
from pymloc.model.sensitivities.boundary_dae import BVPSensitivities
from pymloc.model.variables import InputStateVariables
from pymloc.model.variables import NullVariables
from pymloc.model.variables import ParameterContainer
from pymloc.model.variables.container import InputOutputStateVariables
from pymloc.model.variables.container import StateVariablesContainer
from pymloc.model.variables.time_function import StateVariables
from pymloc.model.variables.time_function import Time
from pymloc.solvers.dynamical_systems.adjoint_sensitivities import AdjointSensitivitiesSolver

np.set_printoptions(precision=4)


@pytest.fixture
def variables3():
    loc_vars = InputStateVariables(2, 1, time=Time(0., 2.))
    hl_vars = ParameterContainer(2, domain=RNDomain(1))
    return hl_vars, loc_vars


@pytest.fixture(params=[0., 1.])
def param_control_3(variables3, request):
    fix_p = request.param
    ll_vars = NullVariables()

    @jax.jit
    def e(p, t):
        q = p[1]
        return jnp.array([[1., 0.], [q, 0.]])

    @jax.jit
    def a(p, t):
        q = p[1]
        return jnp.array([[-1., 0.], [-q, 1.]])

    @jax.jit
    def b(p, t):
        q = p[1]
        return jnp.array([[1.], [q]])

    @jax.jit
    def c(p, t):
        return jnp.identity(2)

    @jax.jit
    def d(p, t):
        return jnp.array([[0.]])

    @jax.jit
    def f(p, t):
        q_0 = p[0]
        q = p[1]
        return jnp.array([1., q + fix_p + q_0])

    return LinearParameterControlSystem(ll_vars, *variables3, e, a, b, c, d, f)


@pytest.fixture
def initial_value_3():
    def ivp(p):
        q = p[1]
        return np.array([2., 0.])

    return ivp


@pytest.fixture
def q_3():
    def q(p, t):
        return jnp.array([[3., 0.], [0., 0.]])

    return q


@pytest.fixture
def s_3():
    def s(p, t):
        return np.zeros((2, 1))

    return s


@pytest.fixture
def m_3():
    def m(p):
        return jnp.zeros((2, 2))

    return m


@pytest.fixture
def pdoc_objective_3(q_3, s_3, r, m_3, variables3):
    time = Time(0., 2.)
    return ParameterLQRObjective(*variables3, time, q_3, s_3, r, m_3)


@pytest.fixture
def pdoc_constraint_3(variables3, param_control_3, initial_value_3):
    return ParameterLQRConstraint(*variables3, param_control_3,
                                  initial_value_3)


@pytest.fixture
def pdoc_object_3(variables3, pdoc_objective_3, pdoc_constraint_3):
    return ParameterDependentOptimalControl(*variables3, pdoc_objective_3,
                                            pdoc_constraint_3)


#Next example


@pytest.fixture
def param_control_2(variables3):

    ll_vars = NullVariables()

    @jax.jit
    def e(p, t):
        q = p[1]
        return jnp.array([[1., 0.], [q, 0.]])

    @jax.jit
    def a(p, t):
        q = p[1]
        return jnp.array([[-1., 0.], [-q, 1.]])

    @jax.jit
    def b(p, t):
        q = p[1]
        return jnp.array([[1.], [q]])

    @jax.jit
    def c(p, t):
        return jnp.identity(2)

    @jax.jit
    def d(p, t):
        return np.array([[0.]])

    @jax.jit
    def f(p, t):
        return np.array([0., 0.])

    return LinearParameterControlSystem(ll_vars, *variables3, e, a, b, c, d, f)


@pytest.fixture
def initial_value_2():
    def ivp(p):
        q = p[1]
        return np.array([2., 0.])

    return ivp


@pytest.fixture
def q_2():
    def q(p, t):
        return jnp.array([[p[0]**2. - 1., 0.], [0., 0.]])

    return q


@pytest.fixture
def s_2():
    def s(p, t):
        return np.zeros((2, 1))

    return s


@pytest.fixture
def m_2():
    def m(p):
        return jnp.zeros((2, 2))

    return m


@pytest.fixture
def pdoc_objective_2(q_2, s_2, r, m_2, variables3):
    time = Time(0., 2.)
    return ParameterLQRObjective(*variables3, time, q_2, s_2, r, m_2)


@pytest.fixture
def pdoc_constraint_2(variables3, param_control_2, initial_value_2):
    return ParameterLQRConstraint(*variables3, param_control_2,
                                  initial_value_2)


@pytest.fixture
def pdoc_object_2(variables3, pdoc_objective_2, pdoc_constraint_2):
    return ParameterDependentOptimalControl(*variables3, pdoc_objective_2,
                                            pdoc_constraint_2)


#Next example
@pytest.fixture
def variables2():
    loc_vars = InputStateVariables(1, 1, time=Time(0., 2.))
    hl_vars = ParameterContainer(1, domain=RNDomain(1))
    return hl_vars, loc_vars


@pytest.fixture
def param_control(variables2):

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

    return LinearParameterControlSystem(ll_vars, *variables2, e, a, b, c, d, f)


@pytest.fixture
def initial_value():
    def ivp(p):
        return np.array([2.])

    return ivp


@pytest.fixture
def q():
    def q(p, t):
        return jnp.array([[p**2. - 1.]])

    return q


@pytest.fixture
def s():
    def s(p, t):
        return np.zeros((1, 1))

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

    return m


@pytest.fixture
def pdoc_objective(q, s, r, m, variables2):
    time = Time(0., 2.)
    return ParameterLQRObjective(*variables2, time, q, s, r, m)


@pytest.fixture
def pdoc_constraint(variables2, param_control, initial_value):
    return ParameterLQRConstraint(*variables2, param_control, initial_value)


@pytest.fixture
def pdoc_constraint(variables2, param_control, initial_value):
    return ParameterLQRConstraint(*variables2, param_control, initial_value)


@pytest.fixture
def pdoc_object(variables2, pdoc_objective, pdoc_constraint):
    return ParameterDependentOptimalControl(*variables2, pdoc_objective,
                                            pdoc_constraint)


#Nonlinear Least Squres fixtures
@pytest.fixture
def f_nlsq():
    def f(ll_vars, hl_vars, loc_vars):
        a, b = hl_vars
        x1, x2 = loc_vars
        f1 = jnp.sqrt(2) * (a - x1)
        f2 = jnp.sqrt(2 * b) * (x2 - x1**2)
        return jnp.array([f1, f2])

    return f


@pytest.fixture
def nllq(variables, objective_nllq):

    return NonLinearLeastSquares(objective_nllq, *variables)


@pytest.fixture()
def objective_nllq(variables, f_nlsq):
    objective = NLLSObjective(*variables, f_nlsq)
    return objective


@pytest.fixture
def bvp_sens_object(linear_param_bvp):
    return BVPSensitivities(linear_param_bvp, n_param=1)


@pytest.fixture
def sens_solver(bvp_sens_object):
    return AdjointSensitivitiesSolver(bvp_sens_object,
                                      rel_tol=1e-1,
                                      abs_tol=1e-1)


@pytest.fixture(params=np.arange(0.1, 2.5, 1.1))
def localized_bvp(bvp_sens_object, request):
    parameters = np.array([request.param])
    return bvp_sens_object.get_sensitivity_bvp(parameters)


@pytest.fixture
def localized_flow_prob(localized_bvp):
    time_interval = Time(0., 1.)
    flowprob = LinearFlow(time_interval, localized_bvp.dynamical_system)
    return flowprob


@pytest.fixture
def localized_ivp(localized_bvp):
    time_interval = Time(0., 1.)
    ivp = InitialValueProblem(np.zeros(2), time_interval,
                              localized_bvp.dynamical_system)
    return ivp


#param dae
@pytest.fixture
def linear_param_bvp(linear_param_dae, param_vars):
    initial_value = lambda p: jnp.array([p, 2., 1.])
    t = Time(0., 1., time_grid=np.linspace(0., 1., 10))
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
