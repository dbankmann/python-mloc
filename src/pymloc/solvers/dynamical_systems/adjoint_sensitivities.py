#
# Copyright (c) 2019-2020
#
# @author: Daniel Bankmann
# @company: Technische Universität Berlin
#
# This file is part of the python package pymloc
# (see https://gitlab.tubit.tu-berlin.de/bankmann91/python-mloc )
#
# License: 3-clause BSD, see https://opensource.org/licenses/BSD-3-Clause
#
import logging
from copy import deepcopy

import numpy as np
import scipy
import scipy.linalg as linalg

from pymloc.model.dynamical_system.flow_problem import LinearFlow

from ...misc import restack
from ...misc import unstack
from ...model.dynamical_system.boundary_value_problem import MultipleBoundaryValueProblem
from ...model.dynamical_system.boundary_value_problem import MultipleBoundaryValues
from ...model.dynamical_system.initial_value_problem import InitialValueProblem
from ...model.dynamical_system.representations import LinearFlowRepresentation
from ...model.sensitivities.boundary_dae import BVPSensitivities
from ...model.variables.container import StateVariablesContainer
from ...model.variables.time_function import Time
from ...solver_container import solver_container_factory
from ..base_solver import TimeSolution
from .sensitivities import SensInhomProjectionNoSubset
from .sensitivities import SensitivitiesSolver

logger = logging.getLogger(__name__)
logging.getLogger("pymloc.solvers.dynamical_systems.pygelda").setLevel(
    logging.WARNING)


class AdjointSensitivitiesSolver(SensitivitiesSolver):
    """Subsolver for computing sensitivities via the adjoint method.
    """
    capital_f_default_class = SensInhomProjectionNoSubset

    def _compute_adjoint_boundary_values(
            self, localized_bvp: MultipleBoundaryValueProblem):
        rank = localized_bvp.dynamical_system.rank
        t_0 = localized_bvp.time_interval.t_0
        t_f = localized_bvp.time_interval.t_f
        t20 = localized_bvp.dynamical_system.t2(t_0)
        t2f = localized_bvp.dynamical_system.t2(t_f)
        z_gamma = localized_bvp.boundary_values.z_gamma
        gamma_0, gamma_f = localized_bvp.boundary_values.boundary_values.transpose(
            2, 0, 1)
        temp = z_gamma.T @ np.block([-gamma_0 @ t20, gamma_f @ t2f])
        self._xi_part: np.ndarray = z_gamma @ self._get_xi_small_inverse_part(
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
        # TODO: QR
        return np.linalg.solve(small_gammas @ small_gammas.T, small_gammas)

    def _setup_adjoint_sensitivity_bvp(self, localized_bvp, parameters, tau):
        n = self._dynamical_system.nn
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
        self._sel_times_projector = gamma_new.T
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
            gamma = np.block([[np.zeros(gamma_new.shape)], [gamma_new]])

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
            return np.zeros((nf, sel_shape[0]))

        time = self._time_interval
        if time.at_bound(tau):
            e_dyn = e_check
            a_dyn = a_check
        else:
            e_dyn = e
            a_dyn = a
        return e_dyn, a_dyn, f

    def _get_adjoint_sens_dae(self, e_check, a_check, n, sel_shape, tau):
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
        # TODO: Make attribute
        nodes = self._get_adjoint_nodes(tau)
        stepsize = self.abs_tol**(1 / 3)

        adjoint_bvp.init_solver(flow_problem,
                                iv_problem,
                                nodes,
                                stepsize,
                                abs_tol=self.abs_tol,
                                rel_tol=self.rel_tol)
        logger.info("Solving adjoint boundary value problem...")
        adjoint_sol_blown_up = adjoint_bvp.solve(time, dynamic_update=True)
        coll_sol = self._adjoint_collapse_solution(adjoint_sol_blown_up[0],
                                                   tau)

        adjoint_sol_blown_up[0](1.31)
        coll_sol.dynamic_update = self._collapsed_dynamic_update(
            adjoint_sol_blown_up[0], tau)

        return coll_sol

    def _collapsed_dynamic_update(self, adjoint_sol, tau):
        def _collapse_dynamic_update(sol, t):
            adjoint_solution = adjoint_sol(
                t[0])  # TODO: Generalize for multiple time points
            time = self._time_interval
            if time.at_bound(tau):
                return adjoint_solution
            else:
                n = self._nn
                if t >= tau:
                    return adjoint_solution[n:]
                else:
                    return adjoint_solution[:n]

        return _collapse_dynamic_update

    def _adjoint_collapse_solution(self, adjoint_solution, tau):
        time = self._time_interval
        if time.at_bound(tau):
            return deepcopy(adjoint_solution)
        else:
            grid = adjoint_solution.time_grid
            solution = adjoint_solution.solution
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
            e0.T @ adjoint_solution.solution[..., 0] + incr_0.T
        ], [ef.T @ adjoint_solution.solution[..., -1] + incr_f.T]])

        return xi

    def _adjoint_integrand(self, t, y, adjoint_solution, f_tilde):
        adj = adjoint_solution(t)
        f_eval = f_tilde(t)
        return unstack(adj.T @ f_eval)

    def _compute_sensitivity(self, capital_f_theta, capital_f_tilde,
                             localized_bvp, solution, adjoint_solution,
                             eplus_e_theta, parameters, tau):

        time = self._time_interval
        t0 = time.t_0
        tf = time.t_f
        selector = self._bvp_param.selector(parameters)
        selector_theta = self._bvp_param.selector_theta(parameters)

        summand1 = np.einsum('ijk, j->ik', selector_theta, solution(tau))
        summand2 = self._capital_fs_instance.summand_2(tau)
        summand3 = selector @ self._capital_fs_instance.f_a_theta(tau)
        xi = self._get_xi(localized_bvp, adjoint_solution, tau)
        summand4 = np.einsum(
            'ij,ik->jk', xi,
            self._boundary_values.get_inhomogeneity_theta(
                solution, parameters))
        summand5 = self._compute_temp5_quant(
            localized_bvp, adjoint_solution,
            eplus_e_theta, solution, tf) - self._compute_temp5_quant(
                localized_bvp, adjoint_solution, eplus_e_theta, solution, t0)
        summand6 = self._compute_temp4_integral(adjoint_solution,
                                                capital_f_tilde, tau)

        logger.info("All summands:\n{}".format(
            np.array(
                [summand1, summand2, summand3, summand4, summand5, summand6])))
        return summand1 + summand2 - summand3 - summand4 - summand5 + summand6

    def _compute_temp4_integral(self, adjoint_solution, capital_f_tilde, tau):
        time = self._time_interval
        t0 = time.t_0
        tf = time.t_f
        shapetemp4 = self._adjoint_sol_shape
        if time.at_bound(tau):
            temp4 = scipy.integrate.solve_ivp(self._adjoint_integrand,
                                              [t0, tf],
                                              np.zeros(shapetemp4).ravel(),
                                              t_eval=[tf],
                                              args=(adjoint_solution,
                                                    capital_f_tilde),
                                              rtol=self.rel_tol,
                                              atol=self.abs_tol)
        else:
            temp4 = scipy.integrate.solve_ivp(self._adjoint_integrand,
                                              [t0, tau],
                                              np.zeros(shapetemp4).ravel(),
                                              t_eval=[tau],
                                              args=(adjoint_solution,
                                                    capital_f_tilde),
                                              rtol=self.rel_tol,
                                              atol=self.abs_tol)

            temp4 = scipy.integrate.solve_ivp(self._adjoint_integrand,
                                              [tau, tf],
                                              temp4.y[..., -1],
                                              t_eval=[tf],
                                              args=(adjoint_solution,
                                                    capital_f_tilde),
                                              rtol=self.rel_tol,
                                              atol=self.abs_tol)
        temp4 = restack(temp4.y[..., -1], shapetemp4)
        return temp4

    def _compute_temp5_quant(self, localized_bvp, adjoint_solution,
                             eplus_e_theta, solution, t):
        e = localized_bvp.dynamical_system.e
        val = np.einsum('ij, jkp,k->ip',
                        adjoint_solution(t).T @ e(t), eplus_e_theta(t),
                        solution(t))
        return val

    def _compute_single_sensitivity(self, tau, localized_bvp, parameters):
        time = deepcopy(localized_bvp.time_interval)
        stepsize = self.abs_tol**(1 / 3)  # Heuristics
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
        solution, node_solution = localized_bvp.solve(time,
                                                      dynamic_update=True)
        capital_f_theta, capital_f_tilde, epluse_theta = self._get_capital_fs(
            localized_bvp, solution, parameters)
        adjoint_solution = self._get_adjoint_solution(localized_bvp,
                                                      parameters, tau, time)
        sensitivity = self._compute_sensitivity(capital_f_theta,
                                                capital_f_tilde, localized_bvp,
                                                solution, adjoint_solution,
                                                epluse_theta, parameters, tau)
        self._solver_params["time_grid"] = adjoint_solution.time_grid
        return sensitivity

    def _run(self, parameters=None, tau=None):
        self.abs_tol /= 6.
        self.rel_tol /= 6.
        if parameters is None:
            parameters = self._bvp_param.parameters.current_values
        localized_bvp = self._bvp_param.get_sensitivity_bvp(parameters)

        if tau is None:
            tau = self._bvp_param.time_interval
        elif isinstance(tau, float):
            tau = Time(tau, tau, time_grid=np.array([tau]))

        gridsize = tau.grid.size
        selshape = self._bvp_param.selector(parameters).shape[0]
        self._adjoint_sol_shape = (selshape, self._bvp_param.n_param)
        solution = np.zeros((*self._adjoint_sol_shape, gridsize))
        for i, tau_val in enumerate(tau.grid):
            logger.info("Compute sensitivity at tau = {}".format(tau_val))
            solution[..., i] = self._compute_single_sensitivity(
                tau_val, localized_bvp, parameters)

        self.abs_tol *= 6
        self.rel_tol *= 6
        return TimeSolution(tau.grid, solution, params=self._solver_params)


solver_container_factory.register_solver(BVPSensitivities,
                                         AdjointSensitivitiesSolver,
                                         default=True)
