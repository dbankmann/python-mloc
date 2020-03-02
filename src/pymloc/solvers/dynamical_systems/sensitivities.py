import numpy as np
import scipy
import scipy.linalg as linalg

from pymloc.model.dynamical_system.flow_problem import LinearFlow

from ...model.dynamical_system.boundary_value_problem import MultipleBoundaryValueProblem
from ...model.dynamical_system.boundary_value_problem import MultipleBoundaryValues
from ...model.dynamical_system.representations import LinearFlowRepresentation
from ...model.sensitivities.boundary_dae import BVPSensitivities
from ...model.variables.container import StateVariablesContainer
from ...model.variables.time_function import Time
from ...solver_container import solver_container_factory
from ..base_solver import BaseSolver


class SensitivitiesSolver(BaseSolver):
    def __init__(self, bvp_param, stepsize):
        if not isinstance(bvp_param, BVPSensitivities):
            raise TypeError(bvp_param)
        self._bvp_param = bvp_param
        self._dynamical_system = bvp_param.dynamical_system
        self._nn = self._dynamical_system.nn
        super().__init__(stepsize)

    def _compute_adjoint_boundary_values(self, localized_bvp):
        n = self._dynamical_system.nn
        rank = localized_bvp.dynamical_system.rank
        t_0 = localized_bvp.time_interval.t_0
        t_f = localized_bvp.time_interval.t_f
        t20 = localized_bvp.dynamical_system.t2(t_0)
        t2f = localized_bvp.dynamical_system.t2(t_f)
        z_gamma = localized_bvp.boundary_values.z_gamma
        gamma_0, gamma_f = localized_bvp.boundary_values.boundary_values.transpose(
            2, 0, 1)
        temp = z_gamma.T @ np.block([gamma_0 @ t20, gamma_f @ t2f])
        small_new_gammas = linalg.null_space(temp)
        gamma_check_0 = z_gamma @ small_new_gammas[:rank, :].T @ t20.T
        gamma_check_f = z_gamma @ small_new_gammas[rank:, :].T @ t2f.T
        return gamma_check_0, gamma_check_f

    def _get_adjoint_dae_coeffs(self, localized_bvp):
        dyn = localized_bvp.dynamical_system

        def e_check(t):
            return -dyn.e(t).T

        def a_check(t):
            return dyn.a(t).T + dyn.der_e(t).T

        return e_check, a_check

    def _get_xi_small_inverse_part(self, small_gammas):
        #TODO: QR
        return np.solve(small_gammas @ small_gammas.T, small_gammas)

    def _get_capital_f_tilde(self, localized_bvp, solution):
        a_dif = self._dynamical_system.a_theta
        e_dif = self._dynamical_system.e_theta
        f_dif = self._dynamical_system.f_theta
        x_d = localized_bvp.dynamical_system.x_d(solution)
        x_d_dot = np.einsum('ijk,jlk->ilk', localized_bvp.dynamical_system.d_d,
                            x_d)

        f_tilde = np.einsum('ijp,j->ip', a_dif, solution) - np.einsum(
            'ijp,j->ip', e_dif, x_d_dot) + f_dif
        return f_tilde

    def _setup_adjoint_sensitivity_bvp(self, localized_bvp, tau):
        n = self._dynamical_system.nn
        n_param = self._bvp_param.n_param
        t_0 = self._bvp_param.boundary_value_problem.time_interval.t_0
        t_f = self._bvp_param.boundary_value_problem.time_interval.t_f
        e_check, a_check = self._get_adjoint_dae_coeffs(localized_bvp)
        sel = self._bvp_param.selector

        def e(t):
            e = e_check(t)
            return linalg.block_diag(e, e)

        def a(t):
            a = a_check(t)
            return linalg.block_diag(a, a)

        def f(t):
            return np.zeros((2 * n, n_param))

        variables = StateVariablesContainer((2 * n, sel.shape[0]))
        adjoint_dyn_sy = LinearFlowRepresentation(variables, e, a, f, 2 * n)

        gamma_0, gamma_f = self._compute_adjoint_boundary_values(localized_bvp)
        zeros = np.zeros((n, n))
        bound_0 = linalg.block_diag(gamma_0 @ e_check(t_0), zeros)
        bound_f = np.block([[zeros, gamma_f @ e_check(t_f)], [zeros, zeros]])
        bound_tau = np.block([[zeros, zeros], [-e_check(tau), e_check(tau)]])
        gamma = np.block(
            [[zeros],
             [(sel @ localized_bvp.dynamical_system.cal_projection(tau)).T]])
        z_gamma = localized_bvp.boundary_values.z_gamma
        z_gamma_new = linalg.block_diag(z_gamma, z_gamma)
        boundary_values = MultipleBoundaryValues((bound_0, bound_tau, bound_f),
                                                 gamma, z_gamma_new)
        t1 = Time(t_0, tau)
        t2 = Time(tau, t_f)
        return MultipleBoundaryValueProblem((t1, t2), adjoint_dyn_sy,
                                            boundary_values)

    def _evaluate_sensitivity_function(self):
        pass

    def _get_adjoint_solution(self, localized_bvp, tau):
        adjoint_bvp = self._setup_adjoint_sensitivity_bvp(localized_bvp, tau)
        flow_problem = self._get_adjoint_flow_problem(adjoint_bvp)
        time = self._bvp_param.boundary_value_problem.time_interval
        #TODO: Make attribute
        intermediate = 5
        nodes = np.concatenate((np.linspace(time.t_0, tau, intermediate),
                                np.linspace(tau, time.t_f, intermediate)[1:]))
        stepsize = 1e-4
        adjoint_bvp.init_solver(flow_problem, nodes, stepsize)
        return adjoint_bvp.solve()

    def _get_adjoint_flow_problem(self, adjoint_bvp):
        time = self._bvp_param.boundary_value_problem.time_interval
        flow_problem = LinearFlow(time, adjoint_bvp.dynamical_system)
        return flow_problem

    def _compute_sensitivity(self, f_tilde, solution, adjoint_solution):
        temp1 = np.einsum('ijk, j->ik', self._bvp_param.selector_theta,
                          solution)

        temp2 = self._bvp_param.selector @ localized_bvp.d_a(
            tau) @ self._bvp_param.f_theta
        temp3 = xi @ self._bvp_param.gamma_theta

        temp4 = scipy.integrate(t0, tf, adjoint_solution.T @ f_tilde)

        return temp1 - temp2 - temp3 - temp4

    def run(self, parameters, tau):
        localized_bvp = self._bvp_param.get_sensitivity_bvp(parameters)
        solution = localized_bvp.solve()
        f_tilde = self._get_capital_f_tilde(localized_bvp, solution)
        adjoint_solution = self._get_adjoint_solution(localized_bvp, tau)
        self._compute_sensitivity(f_tilde, adjoint_solution)


solver_container_factory.register_solver(BVPSensitivities,
                                         SensitivitiesSolver,
                                         default=True)
