import numpy as np
from pygelda.pygelda import Gelda

from ...model.dynamical_system.initial_value_problem import InitialValueProblem
from ...solver_container.factory import solver_container_factory
from ..base_solver import BaseSolver


class PyGELDA(BaseSolver):
    def __init__(self, model, stepsize, *args, **kwargs):
        super().__init__(model, *args, **kwargs)

        def edif(t, ndif):
            return model.dynamical_system.e(t)

        def adif(t, ndif):
            return model.dynamical_system.a(t)

        def fdif(t, ndif):
            return model.dynamical_system.f(t)

        self._nn = self.model.nn
        self._gelda_instance = Gelda(edif, adif, fdif, neq=self._nn, ndif=0)
        self.x0 = model.initial_value
        self.stepsize = stepsize

    def run(self, t0=None, tf=None, x0=None):
        if t0 is None:
            t0 = self.model.initial_time
        if tf is None:
            tf = self.model.final_time
        if x0 is None:
            x0 = self.x0
        times = np.arange(t0, tf + self.stepsize, self.stepsize)

        xout, ierr = self._gelda_instance.solve(times,
                                                x0,
                                                rtol=self.rel_tol,
                                                atol=self.abs_tol)
        if ierr < 0:
            raise ValueError("Simulation did not complete")
        return xout


solver_container_factory.register_solver(InitialValueProblem,
                                         PyGELDA,
                                         default=True)
