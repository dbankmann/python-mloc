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

from pymloc.model.dynamical_system.flow_problem import LinearFlow

from ...model.dynamical_system.boundary_value_problem import BoundaryValueProblem
from ...model.dynamical_system.initial_value_problem import InitialValueProblem
from ...model.dynamical_system.representations import LinearFlowRepresentation
from ...model.sensitivities.boundary_dae import BVPSensitivities
from ...model.variables.container import StateVariablesContainer
from ...solver_container import solver_container_factory
from ..base_solver import TimeSolution
from .sensitivities import SensInhomProjection
from .sensitivities import SensitivitiesSolver

logger = logging.getLogger(__name__)


class ForwardSensitivitiesSolver(SensitivitiesSolver):
    capital_f_default_class = SensInhomProjection

    def _get_forward_system(self, localized_bvp: BoundaryValueProblem,
                            parameters: np.ndarray, f_tilde,
                            solution: TimeSolution):
        dyn_sys = localized_bvp.dynamical_system

        n = self._nn
        n_param = self._bvp_param.n_param
        variables = StateVariablesContainer((n, n_param))
        lin_flow = LinearFlowRepresentation(variables, dyn_sys.e, dyn_sys.a,
                                            f_tilde, dyn_sys.nn)
        bvs = deepcopy(localized_bvp.boundary_values)
        bound_values = self._bvp_param.boundary_value_problem.boundary_values
        bvs.inhomogeneity = bound_values.get_inhomogeneity_theta(
            solution, parameters)

        sens_bvp = BoundaryValueProblem(self._bvp_param.time_interval,
                                        lin_flow, bvs)
        ivp = InitialValueProblem(bvs.inhomogeneity, self._time_interval,
                                  lin_flow)
        return sens_bvp, lin_flow, ivp

    def _compute_sensitivity(self, localized_bvp, parameters, f_tilde,
                             solution):
        forward_bvp, forward_flow, forward_ivp = self._get_forward_system(
            localized_bvp, parameters, f_tilde, solution)
        t0 = self._time_interval.t_0
        tf = self._time_interval.t_f
        time = deepcopy(self._time_interval)
        lin_flow = LinearFlow(time, forward_flow)
        shooting_nodes = np.linspace(t0, tf, 3)
        stepsize = 1e-1
        forward_bvp.init_solver(lin_flow,
                                forward_ivp,
                                shooting_nodes,
                                stepsize=stepsize)
        sol = forward_bvp.solve(time)
        return sol

    def _run(self, parameters):
        localized_bvp = self._bvp_param.get_sensitivity_bvp(parameters)
        time = deepcopy(localized_bvp.time_interval)
        # TODO: Refactor
        stepsize = 1e-1  # needed for integration. TODO: Better way to choose that stepsize?
        time.grid = np.arange(time.t_0, time.t_f, stepsize)
        self._time_interval = time
        flow_prob = LinearFlow(deepcopy(time), localized_bvp.dynamical_system)
        ivp_prob = InitialValueProblem(
            np.zeros((localized_bvp.dynamical_system.nn, )), time,
            localized_bvp.dynamical_system)
        nodes = np.linspace(time.t_0, time.t_f, 3)
        localized_bvp.init_solver(flow_prob,
                                  ivp_prob,
                                  nodes,
                                  stepsize,
                                  abs_tol=self.abs_tol,
                                  rel_tol=self.rel_tol)
        solution = localized_bvp.solve(time)
        solution.interpolation = True
        f_tilde = self._get_capital_fs(localized_bvp, solution, parameters)[0]
        sensitivity = self._compute_sensitivity(localized_bvp, parameters,
                                                f_tilde, solution)
        return sensitivity


solver_container_factory.register_solver(BVPSensitivities,
                                         ForwardSensitivitiesSolver,
                                         default=False)
