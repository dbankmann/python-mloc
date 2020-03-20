import numpy as np
from pygelda.pygelda import Gelda

from ...model.dynamical_system.initial_value_problem import InitialValueProblem
from ...solver_container.factory import solver_container_factory
from ..base_solver import BaseSolver
from ..base_solver import TimeSolution


class PyGELDA(BaseSolver):
    def __init__(self, model, stepsize, f_columns=1, **kwargs):
        super().__init__(model, **kwargs)
        self._f_columns = f_columns

        def edif(t, ndif):
            return model.dynamical_system.e(t)

        def adif(t, ndif):
            return model.dynamical_system.a(t)

        def ith_fdif(i):
            def fdif(t, ndif):
                return np.atleast_2d(model.dynamical_system.f(t))[:, i]

            return fdif

        self._nn = self.model.nn
        self._gelda_instances = [
            Gelda(edif, adif, ith_fdif(i), neq=self._nn, ndif=0)
            for i in range(f_columns)
        ]
        self.x0 = model.initial_value
        self.stepsize = stepsize

    def _run(self, interval=None, x0=None, stepsize=None, n_steps=None):
        if x0 is None:
            x0 = self.x0
        if stepsize is None:
            stepsize = self.stepsize
        if n_steps is None:
            n_steps = np.int(np.ceil(1 / stepsize))
        if interval is None:
            t0 = self.model.initial_time
            tf = self.model.final_time
            times = np.linspace(t0, tf, n_steps + 1)
        else:
            times = interval.grid
        f_1d = x0.ndim == 1
        x0 = np.atleast_2d(x0.T).T  #make 1d array a column vector
        if len(times) == 1:
            return TimeSolution(times, x0)
        xout = np.zeros((*x0.shape, times.size), order='F')

        if x0.ndim > 2:
            raise ValueError(x0)
        for i, x0_f in enumerate(x0.T):
            xout_f, ierr = self._gelda_instances[i].solve(times,
                                                          x0_f,
                                                          rtol=self.rel_tol,
                                                          atol=self.abs_tol)
            xout[..., i, :] = xout_f
        if f_1d:
            xout = np.squeeze(xout)
        if ierr < 0:
            raise ValueError("Simulation did not complete")

        return TimeSolution(times, xout)


solver_container_factory.register_solver(InitialValueProblem,
                                         PyGELDA,
                                         default=True)
