import numpy as np
from scipy.integrate import ode
from scipy.integrate import trapz

from ...model.dynamical_system.flow_problem import LinearFlow
from ...solver_container import solver_container_factory
from ..base_solver import BaseSolver


class DAEFlow(BaseSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ProjectionDAEFlowIntegrator(DAEFlow):
    def __init__(self, dae_flow_instance, time_interval, stepsize):
        super().__init__(dae_flow_instance)
        self._time_interval = time_interval
        self._stepsize = stepsize
        self._homogeneous_flows = None
        self._nn = dae_flow_instance._nn

    @property
    def stepsize(self):
        return self._stepsize

    @property
    def time_interval(self):
        return self._time_interval

    @time_interval.setter
    def time_interval(self, value):
        self._time_interval = value

    def run(self, x0):
        hom_flow = self.get_homogeneous_flows()
        t0 = self.time_interval[0]
        tf = self.time_interval[-1]
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
        stepsize = self.stepsize
        t0 = self.time_interval.t_0
        tf = self.time_interval.t_f
        time_interval = np.arange(t0, tf + stepsize, stepsize)
        intervals = zip(time_interval, time_interval[1:])
        nflows = len(time_interval) - 1
        flows = np.zeros((n, n, nflows), order='F')
        #TODO: Paralellize
        for i, (t_i, t_ip1) in enumerate(intervals):
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
        integrator = ode(f, jac)
        for i, unit_vector in enumerate(np.identity(n)):
            integrator.set_initial_value(unit_vector, t0)
            flow[:, i] = integrator.integrate(tf)
        return flow


solver_container_factory.register_solver(LinearFlow,
                                         ProjectionDAEFlowIntegrator,
                                         default=True)
