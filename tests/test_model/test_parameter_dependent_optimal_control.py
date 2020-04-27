import jax.numpy as jnp
import numpy as np
import pytest

from pymloc.model.dynamical_system.parameter_dae import jac_jax_reshaped

from .test_optimization.test_optimal_control import compare_sol_ref_sol
from .test_optimization.test_optimal_control import compute_ref_sol


def refsol(theta, t0, tf, t, x01):
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
           (-1 + theta)**2 * (1 - t * (1 + theta) - t0 * (1 + theta) + 2 * tf *
                              (1 + theta)))],
         [(1 / ((-1 + theta + np.exp(2 * (-t0 + tf) * theta) *
                 (1 + theta))**2)) * np.exp(-(t + t0) * theta) * x01 *
          (np.exp(2 * t * theta) * (t - t0) *
           (-1 + theta)**2 + np.exp(-2 * t0 * theta + 4 * tf * theta) *
           (-t + t0) * (1 + theta)**2 + np.exp(2 * tf * theta) *
           (-2 + t + t0 - 2 * tf -
            (t + t0 - 2 * tf) * theta**2) + np.exp(2 * (t - t0 + tf) * theta) *
           (2 - t - t0 + 2 * tf + (t + t0 - 2 * tf) * theta**2))]])
    return refsol


class TestPDOCObject:
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
        parameters = np.array(2.)
        intw = pdoc_object.objective.integral_weights
        jac = jac_jax_reshaped(intw, (2, 2))
        jac_val = jac(parameters, 2.)

    def test_sensitivities(self, pdoc_object):
        sens = pdoc_object.get_sensitivities()
        sens.init_solver(abs_tol=1e-2, rel_tol=1e-2)
        sol = sens.solve(parameters=np.array(2.), tau=1.)
        rsol = refsol(2., 0., 2., 1., 2.)
        ref = np.block([[rsol], [-rsol[0]]])
        assert np.allclose(ref, sol.solution, rtol=1e-9, atol=0.4)

    def test_sensitivities_augmented(self, pdoc_object, pdoc_object_2):
        #local_lq = pdoc_object.get_localized_object(hl_value=2.)
        #comp = compare_sol_ref_sol(local_lq, 2.)
        selector = lambda p: jnp.array([[0.5, 0.], [0.5, 0.], [0., 1.],
                                        [0., 0.], [0., 0.]]).T
        selector_shape = (2, 5)
        sens = pdoc_object_2.get_sensitivities(selector, selector_shape)
        sens.init_solver(abs_tol=1e-6, rel_tol=1e-6)
        sol = sens.solve(parameters=np.array([2., 1.]), tau=1.)(1.)
        rsol = refsol(2., 0., 2., 1., 2.)
        ref = np.block([[rsol[0]], [0], [rsol[1]], [0], [-rsol[0]]])
        ref = np.block([[ref, np.zeros((5, 1))]])
        ref = selector(3.) @ ref
        assert np.allclose(ref, sol, rtol=1e-9, atol=1e-2)

    def test_forward_sensitivities(self, pdoc_object):
        sens = pdoc_object.get_sensitivities()
        solver = sens._available_solvers.solvers[1]
        sens.solver = solver
        sens.init_solver(abs_tol=1e-8, rel_tol=1e-8)
        sol = sens.solve(parameters=np.array(2.))
        rsol = refsol(2., 0., 2., 1., 2.)
        ref = np.block([[rsol], [-rsol[0]]])
        assert np.allclose(ref, sol[0](1.), atol=1e-2, rtol=1e-9)

    def test_forward_boundary_sens(self, pdoc_object):
        sens = pdoc_object.get_sensitivities()
        solver = sens._available_solvers.solvers[1]
        sens.solver = solver
        sens.init_solver(abs_tol=1e-8, rel_tol=1e-8)
        sol = sens.solve(parameters=np.array(2.))
        rsol0 = refsol(2., 0., 2., 0., 2.)
        ref0 = np.block([[rsol0], [-rsol0[0]]])
        rsolf = refsol(2., 0., 2., 2., 2.)
        reff = np.block([[rsolf], [-rsolf[0]]])
        assert np.allclose(ref0, sol[0](0.), rtol=1e-9, atol=1e-1)
        assert np.allclose(reff, sol[0](2.), rtol=1e-9, atol=1e-1)

    def test_boundary_sens1(self, pdoc_object):
        sens = pdoc_object.get_sensitivities()
        sens.init_solver(abs_tol=1e-8, rel_tol=1e-8)
        solf = sens.solve(parameters=np.array(2.), tau=2.)
        rsolf = refsol(2., 0., 2., 2., 2.)
        reff = np.block([[rsolf], [-rsolf[0]]])
        assert np.allclose(reff, solf.solution, rtol=1e-9, atol=1e-1)

    def test_boundary_sens2(self, pdoc_object):
        sens = pdoc_object.get_sensitivities()
        sens.init_solver(abs_tol=1e-8, rel_tol=1e-8)
        solf = sens.solve(parameters=np.array(2.), tau=2.)
        rsolf = refsol(2., 0., 2., 2., 2.)
        reff = np.block([[rsolf], [-rsolf[0]]])
        assert np.allclose(reff, solf.solution, rtol=1e-9, atol=1e-1)
