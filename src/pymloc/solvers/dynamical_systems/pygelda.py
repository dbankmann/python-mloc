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
from typing import Callable
from typing import Optional

import numpy as np
from pygelda.pygelda import Gelda

from ...model.dynamical_system.initial_value_problem import InitialValueProblem
from ...solver_container.factory import solver_container_factory
from ..base_solver import BaseSolver
from ..base_solver import TimeSolution


class PyGELDA(BaseSolver):
    """Solver for the python interface pygelda of the FORTRAN solver GELDA."""
    def __init__(self,
                 model: InitialValueProblem,
                 stepsize: float,
                 f_columns: int = 1,
                 **kwargs):
        """
        Parameters
        ----------
        f_columns: Number of columns of the inhomogeneity f. If > 1, this corresponds to
        matrix valued solutions. This structure is necessary for the computation of solutions
        to the adjoint_sensitivity system in :class:`.adjoint_sensitivities.AdjointSensitivitiesSolver`
        """
        super().__init__(model, **kwargs)
        self._f_columns = f_columns

        def edif(t: float, ndif: int) -> np.ndarray:
            return model.dynamical_system.e(t)

        def adif(t: float, ndif: int) -> np.ndarray:
            return model.dynamical_system.a(t)

        def ith_fdif(i: int) -> Callable:
            def fdif(t, ndif):
                return np.atleast_2d(model.dynamical_system.f(t).T).T[:, i]

            return fdif

        self._nn = self.model.nn
        self._gelda_instances = [
            Gelda(edif, adif, ith_fdif(i), neq=self._nn, ndif=0)
            for i in range(f_columns)
        ]
        self.x0 = model.initial_value
        self.stepsize = stepsize

    @property
    def model(self) -> InitialValueProblem:
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    def _run(  # type: ignore[override] # TODO: fix types and interface
            self,
            interval: Optional[np.ndarray] = None,
            x0: Optional[np.ndarray] = None,
            stepsize: Optional[float] = None,
            n_steps: Optional[int] = None):
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
        if len(times) == 1:
            xout = np.zeros((*x0.shape, times.size), order='F')
            xout[..., 0] = x0
            return TimeSolution(times, xout)
        x0 = np.atleast_2d(x0.T).T  # make 1d array a column vector
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
