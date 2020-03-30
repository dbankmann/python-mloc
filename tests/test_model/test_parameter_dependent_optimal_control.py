import jax.numpy as jnp
import numpy as np
import pytest

from pymloc.model.control_system.parameter_dae import LinearParameterControlSystem
from pymloc.model.domains import RNDomain
from pymloc.model.dynamical_system.parameter_dae import jac_jax_reshaped
from pymloc.model.optimization.parameter_optimal_control import ParameterDependentOptimalControl
from pymloc.model.optimization.parameter_optimal_control import ParameterLQRConstraint
from pymloc.model.optimization.parameter_optimal_control import ParameterLQRObjective
from pymloc.model.variables import InputStateVariables
from pymloc.model.variables import NullVariables
from pymloc.model.variables import ParameterContainer
from pymloc.model.variables.time_function import Time

from .test_optimization.test_optimal_control import compare_sol_ref_sol
from .test_optimization.test_optimal_control import compute_ref_sol


@pytest.fixture
def variables():
    loc_vars = InputStateVariables(1, 1, time=Time(0., 2.))
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
        return np.array([[2.]])

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
def pdoc_objective(q, s, r, m, variables):
    time = Time(0., 2.)
    return ParameterLQRObjective(*variables, time, q, s, r, m)


@pytest.fixture
def pdoc_constraint(variables, param_control, initial_value):
    return ParameterLQRConstraint(*variables, param_control, initial_value)


@pytest.fixture
def pdoc_object(variables, pdoc_objective, pdoc_constraint):
    return ParameterDependentOptimalControl(*variables, pdoc_objective,
                                            pdoc_constraint)


class TestPDOCObject:
    def refsol(self, theta, t0, tf, t, x01):
        refsol = np.array(
            [[(1 / ((-1 + theta + np.exp(2 * (-t0 + tf) * theta) *
                     (1 + theta))**2)) * np.exp(-(t + t0) * theta) * x01 *
              (np.exp(-2 * (t0 - 2 * tf) * theta) * (1 + theta)**2 *
               (1 + t + t0 *
                (-1 + theta) - t * theta) - np.exp(2 * (t - t0 + tf) * theta) *
               (1 + theta)**2 *
               (1 + 2 * tf + t * (-1 + theta) + t0 *
                (-1 + theta) - 2 * tf * theta) - np.exp(2 * t * theta) * 2 *
               (-1 + theta)**2 * (1 + t * (1 + theta) - t0 *
                                  (1 + theta)) + np.exp(2 * tf * theta) *
               (-1 + theta)**2 * (1 - t * (1 + theta) - t0 *
                                  (1 + theta) + 2 * tf * (1 + theta)))],
             [(1 / ((-1 + theta + np.exp(2 * (-t0 + tf) * theta) *
                     (1 + theta))**2)) * np.exp(-(t + t0) * theta) * x01 *
              (np.exp(2 * t * theta) * (t - t0) *
               (-1 + theta)**2 + np.exp(-2 * t0 * theta + 4 * tf * theta) *
               (-t + t0) * (1 + theta)**2 + np.exp(2 * tf * theta) *
               (-2 + t + t0 - 2 * tf -
                (t + t0 - 2 * tf) * theta**2) + np.exp(2 *
                                                       (t - t0 + tf) * theta) *
               (2 - t - t0 + 2 * tf + (t + t0 - 2 * tf) * theta**2))]])
        return refsol

    def test_eval_weights(self, pdoc_object):
        random_p, = pdoc_object.parameters.get_random_values()
        random_x, random_u, random_t = pdoc_object.state_input.get_random_values(
        )
        local_lq = pdoc_object.get_localized_object(hl_value=random_p)
        weights = local_lq.objective.integral_weights(1.)
        final_weight = local_lq.objective.final_weight
        assert final_weight.shape == (1, 1)
        assert weights.shape == (2, 2) and weights[0, 0] == random_p[0]**2 - 1

    def test_solution(self, pdoc_object):
        random_p, = pdoc_object.parameters.get_random_values()
        local_lq = pdoc_object.get_localized_object(hl_value=random_p)
        compare_sol_ref_sol(local_lq, random_p[0])

    def test_q_theta(self, pdoc_object):
        parameters = np.array([2.])
        q = pdoc_object.objective.q
        jac = jac_jax_reshaped(q, (1, 1))
        jac_val = jac(parameters, 2.)

        assert jac_val == np.array([[2 * parameters[0]]])

    def test_int_theta(self, pdoc_object):
        parameters = np.array([2.])
        intw = pdoc_object.objective.integral_weights
        jac = jac_jax_reshaped(intw, (2, 2))
        jac_val = jac(parameters, 2.)

    def test_sensitivities(self, pdoc_object):
        sens = pdoc_object.get_sensitivities()
        sens.init_solver(abs_tol=1e-8, rel_tol=1e-8)
        sol = sens.solve(parameters=np.array([2.]), tau=1.)
        rsol = self.refsol(2., 0., 2., 1., 2.)
        ref = np.block([[rsol], [-rsol[0]]])
        assert np.allclose(ref, sol, rtol=1e-9, atol=1e-1)

    def test_forward_sensitivities(self, pdoc_object):
        sens = pdoc_object.get_sensitivities()
        solver = sens._available_solvers.solvers[1]
        sens.solver = solver
        sens.init_solver(abs_tol=1e-8, rel_tol=1e-8)
        sol = sens.solve(parameters=np.array([2.]))
        rsol = self.refsol(2., 0., 2., 1., 2.)
        ref = np.block([[rsol], [-rsol[0]]])
        assert np.allclose(ref, sol[0](1.), atol=1e-2, rtol=1e-9)

    def test_forward_boundary_sens(self, pdoc_object):
        sens = pdoc_object.get_sensitivities()
        solver = sens._available_solvers.solvers[1]
        sens.solver = solver
        sens.init_solver(abs_tol=1e-8, rel_tol=1e-8)
        sol = sens.solve(parameters=np.array([2.]))
        rsol0 = self.refsol(2., 0., 2., 0., 2.)
        ref0 = np.block([[rsol0], [-rsol0[0]]])
        rsolf = self.refsol(2., 0., 2., 2., 2.)
        reff = np.block([[rsolf], [-rsolf[0]]])
        assert np.allclose(ref0, sol[0](0.), rtol=1e-9, atol=1e-1)
        assert np.allclose(reff, sol[0](2.), rtol=1e-9, atol=1e-1)

    def test_boundary_sens1(self, pdoc_object):
        sens = pdoc_object.get_sensitivities()
        sens.init_solver(abs_tol=1e-8, rel_tol=1e-8)
        solf = sens.solve(parameters=np.array([2.]), tau=2.)
        rsolf = self.refsol(2., 0., 2., 2., 2.)
        reff = np.block([[rsolf], [-rsolf[0]]])
        assert np.allclose(reff, solf, rtol=1e-9, atol=1e-1)

    def test_boundary_sens2(self, pdoc_object):
        sens = pdoc_object.get_sensitivities()
        sens.init_solver(abs_tol=1e-8, rel_tol=1e-8)
        solf = sens.solve(parameters=np.array([2.]), tau=2.)
        rsolf = self.refsol(2., 0., 2., 2., 2.)
        reff = np.block([[rsolf], [-rsolf[0]]])
        assert np.allclose(reff, solf, rtol=1e-9, atol=1e-1)
