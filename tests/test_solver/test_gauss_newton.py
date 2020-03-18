import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pymloc.solvers.nonlinear.gauss_newton import GaussNewton


@pytest.fixture(params=np.random.random((100, 2)))
def nllq_obj_loc(nllq, request):
    nllq.higher_level_variables.current_values = request.param
    return nllq.get_localized_object()


@pytest.fixture
def jac(nllq_obj_loc):
    def jacobian(x):
        return np.array(jax.jacobian(nllq_obj_loc.objective.residual)(x))

    return jacobian


@pytest.fixture
def gn_instance(nllq_obj_loc, jac):
    return GaussNewton(nllq_obj_loc, jac, maxiter=10)


class TestGaussNewton:
    def test_solve(self, gn_instance, nllq):
        params = nllq.higher_level_variables.current_values
        solution_obj = gn_instance.run(x0=np.array([3., 4.]))
        sol = solution_obj.solution
        ref_sol = np.array([params[0], params[0]**2])
        assert np.allclose(sol, ref_sol)
