import numpy as np
import pytest

from pymloc.model.dynamical_system.boundary_value_problem import MultipleBoundaryValueProblem
from pymloc.model.dynamical_system.flow_problem import LinearFlow
from pymloc.model.variables.time_function import Time


class TestBVPSensitivitiesSolver:
    def test_compute_adjoint_boundary_values(self, sens_solver, localized_bvp):
        sens_solver._compute_adjoint_boundary_values(localized_bvp)

    def test_get_adjoint_dae_coeffs(self, sens_solver, localized_bvp):
        sens_solver._get_adjoint_dae_coeffs(localized_bvp)

    @pytest.mark.parametrize("tau", np.arange(0.1, 1., 0.1))
    def test_solve_adjoint_dae(self, sens_solver, localized_bvp, tau):
        sol = sens_solver._get_adjoint_solution(localized_bvp, tau)
