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

    @pytest.mark.parametrize("tau", np.arange(0.1, 1., 0.1))
    def test_f_tilde(self, sens_solver, localized_bvp, tau):
        time = localized_bvp.time_interval
        flow_prob = LinearFlow(time, localized_bvp.dynamical_system)
        nodes = np.linspace(time.t_0, time.t_f, 5)
        stepsize = 1e-4
        localized_bvp.init_solver(flow_prob, nodes, stepsize)
        import ipdb
        ipdb.set_trace()
        sens_solver._get_capital_f_tilde(
            localized_bvp, localized_bvp.solve(),
            localized_bvp._localization_parameters[0])

    #def test_compute_sensitivity(self, f_tilde, solution, adjoint_solution):
    @pytest.mark.parametrize("tau", np.arange(0.1, 1., 0.1))
    def test_run(self, sens_solver, localized_bvp, tau):
        sens_solver.run(localized_bvp._localization_parameters[0], tau)