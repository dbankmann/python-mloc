import scipy.linalg as linalg

from ...model.optimization.nonlinear_leastsquares import LocalNonLinearLeastSquares
from ...solver_container import solver_container_factory
from ..base_solver import BaseSolver
from ..base_solver import Solution


class GaussNewton(BaseSolver):
    def __init__(self, nllq, sensitivity_fun=None, max_iter=20):
        if not isinstance(nllq, LocalNonLinearLeastSquares):
            raise TypeError(nllq)
        self._nllq = nllq
        self._jac = sensitivity_fun

        super().__init__(nllq, max_iter=maxiter)
        super().__init__(nllq, max_iter=max_iter)

    def _newton_step(self, x, f):
        jac = self._get_jacobian(x)
        q, r = linalg.qr(jac)
        x_new = x - linalg.solve(r.T @ r, r.T) @ q.T @ f
        return x_new

    def _get_jacobian(self, x):
        return self._jac(x)

    def _run(self, x0):
        x = x0
        for i in range(self.max_iter):
            f = self._nllq.objective.residual(x)
            if not self.abort(f):
                x = self._newton_step(x, f)
            else:
                break
        solver_params = {"iter": i}
        return Solution(x, solver_params)


solver_container_factory.register_solver(LocalNonLinearLeastSquares,
                                         GaussNewton,
                                         default=True)
