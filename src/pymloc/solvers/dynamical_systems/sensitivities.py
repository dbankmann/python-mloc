import logging
from copy import deepcopy

import numpy as np
import scipy
import scipy.linalg as linalg

from pymloc.model.dynamical_system.flow_problem import LinearFlow

from ...model.dynamical_system.boundary_value_problem import MultipleBoundaryValueProblem
from ...model.dynamical_system.boundary_value_problem import MultipleBoundaryValues
from ...model.dynamical_system.initial_value_problem import InitialValueProblem
from ...model.dynamical_system.representations import LinearFlowRepresentation
from ...model.sensitivities.boundary_dae import BVPSensitivities
from ...model.variables.container import StateVariablesContainer
from ...model.variables.time_function import Time
from ...solver_container import solver_container_factory
from ..base_solver import BaseSolver
from ..base_solver import TimeSolution

logger = logging.getLogger(__name__)


class SensitivitiesSolver(BaseSolver):
    def __init__(self, bvp_param, *args, **kwargs):
        if not isinstance(bvp_param, BVPSensitivities):
            raise TypeError(bvp_param)
        self._bvp_param = bvp_param
        self._dynamical_system = bvp_param.dynamical_system
        self._nn = self._dynamical_system.nn
        self._time_interval = self._bvp_param.time_interval
        self._boundary_values = self._bvp_param.boundary_value_problem.boundary_values
        super().__init__(*args, **kwargs)

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
        self._xi_part = z_gamma @ self._get_xi_small_inverse_part(
            temp) @ linalg.block_diag(t20.T, t2f.T)
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
        return np.linalg.solve(small_gammas @ small_gammas.T, small_gammas)

    def _get_capital_f_tilde(self, localized_bvp, solution, parameter):
        def a_dif(t):
            return self._dynamical_system.a_theta(parameter, t)

        def e_dif(t):
            return self._dynamical_system.e_theta(parameter, t)

        def f_dif(t):
            return self._dynamical_system.f_theta(parameter, t)

        def x_d(t):
            return localized_bvp.dynamical_system.x_d(t, solution(t))

        def x_d_dot(t):
            return np.einsum('ij,j->i', localized_bvp.dynamical_system.d_d(t),
                             x_d(t))

        def f_tilde(t):
            f_tilde = np.einsum(
                'ijk,j->ik', a_dif(t), solution(t)) - np.einsum(
                    'ijk,j->ik', e_dif(t), x_d_dot(t)) + f_dif(t)
            return f_tilde

        return f_tilde

    def _setup_adjoint_sensitivity_bvp(self, localized_bvp, parameters, tau):
        n = self._dynamical_system.nn
        n_param = self._bvp_param.n_param
        t_0 = self._bvp_param.boundary_value_problem.time_interval.t_0
        t_f = self._bvp_param.boundary_value_problem.time_interval.t_f
        e_check, a_check = self._get_adjoint_dae_coeffs(localized_bvp)
        sel = self._bvp_param.selector(parameters)

        def e(t):
            e = e_check(t)
            return linalg.block_diag(e, e)

        def a(t):
            a = a_check(t)
            return linalg.block_diag(a, a)

        def f(t):
            return np.zeros((2 * n, sel.shape[1]))

        variables = StateVariablesContainer((2 * n, sel.shape[0]))
        adjoint_dyn_sy = LinearFlowRepresentation(variables, e, a, f, 2 * n)

        gamma_0, gamma_f = self._compute_adjoint_boundary_values(localized_bvp)
        zeros = np.zeros((n, n))
        bound_0 = linalg.block_diag(gamma_0 @ e_check(t_0), zeros)
        bound_tau = np.block([[zeros, zeros], [-e_check(tau), e_check(tau)]])
        bound_f = np.block([[zeros, gamma_f @ e_check(t_f)], [zeros, zeros]])
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

    def _get_adjoint_solution(self, localized_bvp, parameters, tau, time):
        adjoint_bvp = self._setup_adjoint_sensitivity_bvp(
            localized_bvp, parameters, tau)
        flow_problem = self._get_adjoint_flow_problem(adjoint_bvp)
        iv_problem = self._get_adjoint_ivp_problem(adjoint_bvp)
        #TODO: Make attribute
        intermediate = 3
        nodes = np.concatenate((np.linspace(time.t_0, tau, intermediate),
                                np.linspace(tau, time.t_f, intermediate)[1:]))
        stepsize = 1e-1
        adjoint_bvp.init_solver(flow_problem,
                                iv_problem,
                                nodes,
                                stepsize,
                                rel_tol=self.rel_tol,
                                abs_tol=self.abs_tol)
        logger.info("Solving adjoint boundary value problem...")
        adjoint_sol_blown_up = adjoint_bvp.solve(time)[0]
        return self._adjoint_collapse_solution(adjoint_sol_blown_up, tau)

    def _adjoint_collapse_solution(self, adjoint_solution, tau):
        n = self._nn
        grid = adjoint_solution.time_grid
        solution = adjoint_solution.solution
        tsize = grid.size
        sshape = solution.shape

        idx = np.searchsorted(grid, tau)
        newn = sshape[0] // 2
        coll_solution = np.empty((newn, *sshape[1:]))
        coll_solution[:, :idx] = solution[:newn, :idx]
        coll_solution[:, idx:] = solution[newn:, idx:]
        return TimeSolution(grid, coll_solution)

    def _get_adjoint_flow_problem(self, adjoint_bvp):
        time = self._bvp_param.boundary_value_problem.time_interval
        flow_problem = LinearFlow(time, adjoint_bvp.dynamical_system)
        return flow_problem

    def _get_adjoint_ivp_problem(self, adjoint_bvp):
        time = self._bvp_param.boundary_value_problem.time_interval
        initial_value = np.zeros(self._nn)
        ivp = InitialValueProblem(initial_value, time,
                                  adjoint_bvp.dynamical_system)
        return ivp

    def _compute_sensitivity(self, f_tilde, localized_bvp, solution,
                             adjoint_solution, parameters, tau):
        time = self._time_interval
        t0 = time.t_0
        tf = time.t_f

        temp1 = np.einsum('ijk, j->ik',
                          self._bvp_param.selector_theta(parameters),
                          solution(tau))
        temp2 = self._bvp_param.selector(
            parameters) @ localized_bvp.dynamical_system.d_a(
                tau) @ self._bvp_param.dynamical_system.f_theta(
                    parameters, tau)
        xi = self._xi_part @ np.block([[adjoint_solution.solution[..., 0]],
                                       [adjoint_solution.solution[..., -1]]])
        temp3 = np.einsum(
            'ij,ikl->jk', xi,
            self._boundary_values.inhomogeinity_theta(parameters))
        f_tilde_arr = np.array([f_tilde(t) for t in solution.time_grid
                                ])  #TODO: slow on constant coefficients.
        temp4m = adjoint_solution.solution.T @ f_tilde_arr
        temp4 = scipy.integrate.trapz(temp4m.transpose(1, 2, 0),
                                      solution.time_grid)

        return temp1 - temp2 - temp3 - temp4

    def _run(self, parameters, tau):
        localized_bvp = self._bvp_param.get_sensitivity_bvp(parameters)
        time = deepcopy(localized_bvp.time_interval)
        stepsize = 1e-1  #needed for integration. TODO:Better way to choose that stepsize?
        time.grid = np.hstack((np.arange(time.t_0, tau, stepsize),
                               np.arange(tau, time.t_f + stepsize, stepsize)))
        flow_prob = LinearFlow(deepcopy(time), localized_bvp.dynamical_system)
        ivp_prob = InitialValueProblem(
            np.zeros((localized_bvp.dynamical_system.nn, )), time,
            localized_bvp.dynamical_system)
        nodes = np.linspace(time.t_0, time.t_f, 5)
        localized_bvp.init_solver(flow_prob,
                                  ivp_prob,
                                  nodes,
                                  stepsize,
                                  abs_tol=self.abs_tol,
                                  rel_tol=self.rel_tol)
        #time.add_to_grid(tau)
        solution, node_solution = localized_bvp.solve(time)
        f_tilde = self._get_capital_f_tilde(localized_bvp, solution,
                                            parameters)
        adjoint_solution = self._get_adjoint_solution(localized_bvp,
                                                      parameters, tau, time)
        sensitivity = self._compute_sensitivity(f_tilde, localized_bvp,
                                                solution, adjoint_solution,
                                                parameters, tau)


solver_container_factory.register_solver(BVPSensitivities,
                                         SensitivitiesSolver,
                                         default=True)
