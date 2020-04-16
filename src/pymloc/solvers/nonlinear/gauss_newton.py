import logging

import numpy as np
import scipy.linalg as linalg

from ...model.optimization.nonlinear_leastsquares import LocalNonLinearLeastSquares
from ...solver_container import solver_container_factory
from ..base_solver import BaseSolver
from ..base_solver import Solution

logger = logging.getLogger(__name__)


class GaussNewton(BaseSolver):
    def __init__(self,
                 nllq,
                 sensitivity_fun=None,
                 max_iter=20,
                 *args,
                 **kwargs):
        if not isinstance(nllq, LocalNonLinearLeastSquares):
            raise TypeError(nllq)
        self._nllq = nllq
        self._jac = sensitivity_fun
        self._variables = nllq.variables
        self._upper_eta = 1.

        super().__init__(nllq, max_iter=max_iter, *args, **kwargs)

    @property
    def upper_eta(self):
        return self._upper_eta

    @upper_eta.setter
    def upper_eta(self, value):
        self._upper_eta = value

    @property
    def lower_abs_tolerance(self):
        return self._lower_abs_tolerance

    def _get_lower_tolerance(self, jac, res):
        safety_factor = 0.9
        eta = self._upper_eta * safety_factor
        tmp1 = eta * np.linalg.norm(jac.T @ res)
        svals = np.linalg.svd(jac)[1]
        smin = svals[0]
        smax = svals[-1]
        condJ = smax / smin
        tmp2 = (condJ**2 + 2 * eta) * smax
        tmp3 = (condJ**2 + eta * condJ + 1) * np.linalg.norm(res)
        atol = tmp1 / (tmp2 + tmp3)
        self._lower_abs_tolerance = atol
        return atol

    def _newton_step(self, x, f):
        jac = self._get_jacobian(x)
        residual = jac.T @ f
        logger.info("Current residual: {}".format(residual))
        logger.info("Current allowed lower level tolerance: {}".format(
            self._get_lower_tolerance(jac, f)))
        if not self.abort(residual):
            q, r = linalg.qr(jac)
            x_new = x - linalg.solve(r.T @ r, r.T) @ q.T @ f
            logger.info("New x value:\n{}\nNew jac value:\n{}".format(
                x_new, jac))
            return x_new, True
        else:
            return x, False

    def _get_jacobian(self, x):
        if self._jac is None:
            return self._nllq.objective.get_jac(x)
        else:
            return self._jac(x)

    def _get_x0(self, x0):
        if x0 is None:
            x0 = self._variables.current_values
        return x0

    def _run(self, x0=None):
        x = self._get_x0(x0)
        for i in range(self.max_iter):
            logger.info("Starting iteration: {}".format(i))
            f = self._nllq.objective.residual(x)
            x, cont = self._newton_step(x, f)
            if not cont:
                break
            self._variables.current_values = x
        solver_params = {"iter": i}
        return Solution(x, solver_params)


solver_container_factory.register_solver(LocalNonLinearLeastSquares,
                                         GaussNewton,
                                         default=True)
