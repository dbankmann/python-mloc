import logging

import scipy.linalg as linalg

from ...model.optimization.nonlinear_leastsquares import LocalNonLinearLeastSquares
from ...solver_container import solver_container_factory
from ..base_solver import BaseSolver
from ..base_solver import Solution

logger = logging.getLogger(__name__)


class GaussNewton(BaseSolver):
    def __init__(self, nllq, sensitivity_fun=None, max_iter=20):
        if not isinstance(nllq, LocalNonLinearLeastSquares):
            raise TypeError(nllq)
        self._nllq = nllq
        self._jac = sensitivity_fun
        self._variables = nllq.variables

        super().__init__(nllq, max_iter=max_iter)

    def _newton_step(self, x, f):
        jac = self._get_jacobian(x)
        q, r = linalg.qr(jac)
        x_new = x - linalg.solve(r.T @ r, r.T) @ q.T @ f
        logger.info("New x value:\n{}\nNew jac value:\n{}".format(x_new, jac))
        return x_new

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
            logger.info("Current residual: {}".format(f))
            if not self.abort(f):
                x = self._newton_step(x, f)
                self._variables.current_values = x
            else:
                break
        solver_params = {"iter": i}
        return Solution(x, solver_params)


solver_container_factory.register_solver(LocalNonLinearLeastSquares,
                                         GaussNewton,
                                         default=True)
