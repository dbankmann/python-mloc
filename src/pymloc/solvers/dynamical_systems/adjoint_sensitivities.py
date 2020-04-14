import logging
from copy import deepcopy

import numpy as np
import scipy
import scipy.linalg as linalg

from pymloc.model.dynamical_system.flow_problem import LinearFlow

from ...model.dynamical_system.boundary_value_problem import BoundaryValueProblem
from ...model.dynamical_system.boundary_value_problem import BoundaryValues
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
from .sensitivities import SensitivitiesSolver

logger = logging.getLogger(__name__)


class AdjointSensitivitiesSolver(SensitivitiesSolver):
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
        temp = z_gamma.T @ np.block([-gamma_0 @ t20, gamma_f @ t2f])
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

    def _setup_adjoint_sensitivity_bvp(self, localized_bvp, parameters, tau):
        n = self._dynamical_system.nn
        n_param = self._bvp_param.n_param
        t_0 = self._bvp_param.boundary_value_problem.time_interval.t_0
        t_f = self._bvp_param.boundary_value_problem.time_interval.t_f
        e_check, a_check = self._get_adjoint_dae_coeffs(localized_bvp)
        sel = self._bvp_param.selector(parameters)
        adjoint_dyn_sys = self._get_adjoint_sens_dae(e_check, a_check, n,
                                                     sel.shape, tau)
        boundary_values = self._get_adjoint_bvs(localized_bvp, e_check, n, t_0,
                                                t_f, sel, tau)
        times = self._get_adjoint_times(t_0, t_f, tau)
        return MultipleBoundaryValueProblem(times, adjoint_dyn_sys,
                                            boundary_values)

    def _get_adjoint_times(self, t_0, t_f, tau):
        time = self._time_interval
        if time.at_bound(tau):
            times = (Time(t_0, t_f), )
        else:
            t1 = Time(t_0, tau)
            t2 = Time(tau, t_f)
            times = (t1, t2)
        return times

    def _get_adjoint_bvs(self, localized_bvp, e_check, n, t_0, t_f, sel, tau):
        z_gamma_new = self._get_z_gamma_new(localized_bvp, tau)
        bounds, gamma = self._get_adjoint_bound_coeffs(localized_bvp, e_check,
                                                       n, t_0, t_f, sel, tau)
        boundary_values = MultipleBoundaryValues(bounds, gamma, z_gamma_new)
        return boundary_values

    def _get_adjoint_bound_coeffs(self, localized_bvp, e_check, n, t_0, t_f,
                                  sel, tau):
        zeros = np.zeros((n, n))
        time = self._time_interval
        gamma_0, gamma_f = self._compute_adjoint_boundary_values(localized_bvp)

        gamma_new_0 = gamma_0 @ e_check(t_0)
        gamma_new_f = gamma_f @ e_check(t_f)
        gamma_new = (
            sel @ localized_bvp.dynamical_system.cal_projection(tau)).T
        self._sel_times_projector = gamma_new
        if time.at_bound(tau):
            bound_0 = gamma_new_0
            bound_f = gamma_new_f
            bounds = (bound_0, bound_f)
            if time.at_upper_bound(tau):
                gamma = -gamma_f @ gamma_new
            else:
                gamma = gamma_0 @ gamma_new

        else:
            bound_0 = linalg.block_diag(gamma_new_0, zeros)
            bound_tau = np.block([[zeros, zeros],
                                  [-e_check(tau), e_check(tau)]])
            bound_f = np.block([[zeros, gamma_new_f], [zeros, zeros]])
            bounds = (bound_0, bound_tau, bound_f)
            gamma = np.block([[zeros], [gamma_new]])

        return bounds, gamma

    def _get_z_gamma_new(self, localized_bvp, tau):
        z_gamma = localized_bvp.boundary_values.z_gamma
        time = self._time_interval
        if time.at_bound(tau):
            z_gamma_new = z_gamma

        else:
            z_gamma_new = linalg.block_diag(z_gamma, z_gamma)
        return z_gamma_new

    def _get_nf(self, tau, n):
        time = self._time_interval
        if time.at_bound(tau):
            nf = n
        else:
            nf = 2 * n

        return nf

    def _get_adjoint_sens_dae_coeffs(self, tau, e_check, a_check, nf,
                                     sel_shape):
        def e(t):
            e = e_check(t)
            return linalg.block_diag(e, e)

        def a(t):
            a = a_check(t)
            return linalg.block_diag(a, a)

        def f(t):
            return np.zeros((nf, sel_shape[1]))

        time = self._time_interval
        if time.at_bound(tau):
            e_dyn = e_check
            a_dyn = a_check
        else:
            e_dyn = e
            a_dyn = a
        return e_dyn, a_dyn, f

    def _get_adjoint_sens_dae(self, e_check, a_check, n, sel_shape, tau):
        time = self._time_interval
        nf = self._get_nf(tau, n)
        e_dyn, a_dyn, f = self._get_adjoint_sens_dae_coeffs(
            tau, e_check, a_check, nf, sel_shape)
        variables = StateVariablesContainer((nf, sel_shape[0]))
        adjoint_dyn_sys = LinearFlowRepresentation(variables, e_dyn, a_dyn, f,
                                                   nf)
        return adjoint_dyn_sys

    def _evaluate_sensitivity_function(self):
        pass

    def _get_adjoint_nodes(self, tau):
        time = self._time_interval
        intermediate = 2
        if time.at_bound(tau):
            nodes = np.linspace(time.t_0, time.t_f, 2 * intermediate - 1)
        else:
            nodes = np.concatenate((np.linspace(time.t_0, tau, intermediate),
                                    np.linspace(tau, time.t_f,
                                                intermediate)[1:]))
        return nodes

    def _get_adjoint_solution(self, localized_bvp, parameters, tau, time):
        logger.info("Assembling adjoint sensitivity boundary value problem...")
        adjoint_bvp = self._setup_adjoint_sensitivity_bvp(
            localized_bvp, parameters, tau)
        flow_problem = self._get_adjoint_flow_problem(adjoint_bvp)
        iv_problem = self._get_adjoint_ivp_problem(adjoint_bvp)
        #TODO: Make attribute
        nodes = self._get_adjoint_nodes(tau)
        stepsize = 1e-6
        adjoint_bvp.init_solver(flow_problem,
                                iv_problem,
                                nodes,
                                stepsize,
                                rel_tol=self.rel_tol,
                                abs_tol=self.abs_tol)
        logger.info("Solving adjoint boundary value problem...")
        adjoint_sol_blown_up = adjoint_bvp.solve(time)
        coll_sol = self._adjoint_collapse_solution(adjoint_sol_blown_up[0],
                                                   tau)

        return coll_sol

    def _adjoint_collapse_solution(self, adjoint_solution, tau):
        time = self._time_interval
        if time.at_bound(tau):
            return adjoint_solution
        else:
            n = self._nn
            grid = adjoint_solution.time_grid
            solution = adjoint_solution.solution
            tsize = grid.size
            sshape = solution.shape

            idx = np.searchsorted(grid, tau)
            newn = sshape[0] // 2
            coll_solution = np.empty((newn, *sshape[1:]))
            coll_solution[..., :idx] = solution[:newn, ..., :idx]
            coll_solution[..., idx:] = solution[newn:, ..., idx:]
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

    def _get_xi(self, localized_bvp, adjoint_solution, tau):
        time = self._time_interval
        t0 = time.t_0
        tf = time.t_f
        e0 = localized_bvp.dynamical_system.e(t0)
        ef = localized_bvp.dynamical_system.e(tf)
        incr_0 = np.zeros(1)
        incr_f = np.zeros(1)

        if time.at_upper_bound(tau):
            incr_f = self._sel_times_projector
        elif time.at_lower_bound(tau):
            incr_0 = self._sel_times_projector

        xi = self._xi_part @ np.block([[
            e0.T @ adjoint_solution.solution[..., 0] + incr_0
        ], [ef.T @ adjoint_solution.solution[..., -1] + incr_f]])

        return xi

    def _compute_sensitivity(self, f_tilde, localized_bvp, solution,
                             adjoint_solution, parameters, tau):

        temp1 = np.einsum('ijk, j->ik',
                          self._bvp_param.selector_theta(parameters),
                          solution(tau))
        temp2 = self._bvp_param.selector(
            parameters) @ localized_bvp.dynamical_system.d_a(
                tau) @ self._bvp_param.dynamical_system.f_theta(
                    parameters, tau)
        xi = self._get_xi(localized_bvp, adjoint_solution, tau)
        temp3 = np.einsum(
            'ij,ik->jk', xi,
            self._boundary_values.get_inhomogeneity_theta(
                solution, parameters))
        f_tilde_arr = np.array([f_tilde(t) for t in solution.time_grid
                                ])  #TODO: slow on constant coefficients.
        temp4m = adjoint_solution.solution.T @ f_tilde_arr
        temp4 = scipy.integrate.trapz(temp4m.transpose(1, 2, 0),
                                      solution.time_grid)

        return temp1 - temp2 - temp3 + temp4

    def _run(self, parameters, tau):
        localized_bvp = self._bvp_param.get_sensitivity_bvp(parameters)
        time = deepcopy(localized_bvp.time_interval)
        stepsize = 1e-1  #needed for integration. TODO:Better way to choose that stepsize?
        time.grid = np.hstack(
            (np.arange(time.t_0, tau,
                       stepsize), np.arange(tau, time.t_f, stepsize)))
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
        solution, node_solution = localized_bvp.solve(time)
        f_tilde = self._get_capital_f_tilde(localized_bvp, solution,
                                            parameters)
        adjoint_solution = self._get_adjoint_solution(localized_bvp,
                                                      parameters, tau, time)
        sensitivity = self._compute_sensitivity(f_tilde, localized_bvp,
                                                solution, adjoint_solution,
                                                parameters, tau)
        return sensitivity


solver_container_factory.register_solver(BVPSensitivities,
                                         AdjointSensitivitiesSolver,
                                         default=True)