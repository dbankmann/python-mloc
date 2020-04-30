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

import numpy as np
from scipy.integrate import ode
from scipy.integrate import trapz

from ...model.dynamical_system.flow_problem import LinearFlow
from ...solver_container import solver_container_factory
from ..base_solver import BaseSolver

logger = logging.getLogger(__name__)


class DAEFlow(BaseSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ProjectionDAEFlowIntegrator(DAEFlow):
    def __init__(self, dae_flow_instance, time_interval, stepsize, *args,
                 **kwargs):
        super().__init__(dae_flow_instance, *args, **kwargs)
        self._time_interval = time_interval
        self._stepsize = stepsize
        self._homogeneous_flows = None
        self._nn = dae_flow_instance._nn

    @property
    def _intervals(self):
        time_grid = self._time_interval.grid
        return zip(time_grid, time_grid[1:])

    @property
    def stepsize(self):
        return self._stepsize

    @property
    def time_interval(self):
        return self._time_interval

    @time_interval.setter
    def time_interval(self, value):
        self._time_interval = value

    def _run(self, x0):
        hom_flow = self.get_homogeneous_flows()
        t0 = self.time_interval.t_0
        tf = self.time_interval.t_f
        fds = np.zeros((hom_flow.shape[1:]), order='F')
        for i, t in enumerate(self.time_interval):
            fds[:, i] = self.model.flow_dae.f_d(t)

        hom = hom_flow[..., -1] @ x0
        integrand = np.einsum('ijr,jr->ir', hom_flow, fds)
        inhom = trapz(integrand,
                      self.time_interval) - self.model.flow_dae.f_a(tf)
        return hom + inhom

    def get_homogeneous_flows(self):
        if self._homogeneous_flows is None:
            self._save_homogeneous_flows()
        return self._homogeneous_flows

    def _save_homogeneous_flows(self):
        n = self._nn
        time_grid = self.time_interval.grid
        intervals = self._intervals
        nflows = len(time_grid) - 1
        flows = np.zeros((n, n, nflows), order='F')
        #TODO: Paralellize
        for i, (t_i, t_ip1) in enumerate(intervals):
            logger.info("Computing solution in the interval ({}, {})".format(
                t_i, t_ip1))
            flows[:, :, i] = self.homogeneous_flow(t_i, t_ip1)
        self._homogeneous_flows = flows

    def homogeneous_flow(self, t0, tf):
        n = self._nn
        flow = np.zeros((n, n), order='F')

        #TODO: Potentially Slow! implement a flow routine at low level routine or parallelize
        def f(t, x):
            return self.model.flow_dae.d_d(t) @ x

        def jac(t, x):
            return self.model.flow_dae.d_d(t)

        h = self.stepsize
        integrator = ode(f, jac).set_integrator('dopri5',
                                                atol=self.abs_tol,
                                                rtol=self.rel_tol)
        for i, unit_vector in enumerate(np.identity(n)):
            logger.debug("Compute solution for {}-th unit vector".format(i))
            integrator.set_initial_value(unit_vector, t0)
            flow[:, i] = integrator.integrate(tf)
        return flow

    def forward_solve_differential(self, node_values):
        end_values = np.zeros(node_values.shape)
        for i, ((tm, tp),
                values) in enumerate(zip(self._intervals, node_values.T)):
            logger.info(
                "Computing inhomogeneous solution in the interval ({}, {})".
                format(tm, tp))
            end_values[..., i] = self._forward_solve(tm, tp, values)

        return end_values

    def _forward_solve(self, tm, tp, values):
        shape = values.shape
        values2d = np.atleast_2d(values)
        value_arr = np.empty(values2d.shape).T

        def jac(t, x):
            return self.model.flow_dae.d_d(t)

        for i, value in enumerate(values2d):

            def f(t, x):
                return self.model.flow_dae.d_d(t) @ x + np.atleast_2d(
                    self.model.flow_dae.f_d(t).T).T[:, i]

            integrator = ode(f, jac).set_integrator('dopri5',
                                                    atol=self.abs_tol,
                                                    rtol=self.rel_tol)
            integrator.set_initial_value(value, tm)
            value_arr[:, i] = integrator.integrate(tp)
        if len(shape) == 1:
            value_arr = np.squeeze(value_arr, axis=1)
        return value_arr


solver_container_factory.register_solver(LinearFlow,
                                         ProjectionDAEFlowIntegrator,
                                         default=True)
