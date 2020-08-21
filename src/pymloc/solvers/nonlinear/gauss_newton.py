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
        self.save_intermediate = False  # TODO: Generalize to BaseSolver
        self._intermediates = {
            'iteration': [],
            'x': [],
            'r': [],
            'J': [],
            'JTr': [],
            'atol': [],
            'cond': [],
        }

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

    def _save(self, key, value):
        if self.save_intermediate:
            self._intermediates[key].append(value)

    def _get_lower_tolerance(self, jac, res):
        safety_factor = 0.9
        eta = self._upper_eta * safety_factor
        tmp1 = eta * np.linalg.norm(jac.T @ res)
        svals = np.linalg.svd(jac)[1]
        smin = svals[0]
        smax = svals[-1]
        condJ = smax / smin

        tmp2 = (condJ**2 + 2 * eta) * smax
        resnorm = np.linalg.norm(res)
        tmp3 = (condJ**2 + eta * condJ + 1) * resnorm
        atol = tmp1 / (tmp2 + tmp3)
        self._lower_abs_tolerance = atol
        logger.info("Current ||r||_2: {}".format(resnorm))
        self._save('r', resnorm)
        logger.info("Current ||J||_2: {}".format(smax))
        self._save('J', smax)
        logger.info("Current cond(J): {}".format(condJ))
        self._save('cond', condJ)
        self._save('atol', atol)
        return atol

    def _newton_step(self, x, f):
        jac = self._get_jacobian(x)
        residual = jac.T @ f
        residualnorm = np.linalg.norm(residual)
        logger.info("Current residual ||J^T r||_2: {}".format(residualnorm))
        self._save('JTr', residualnorm)
        logger.info("Current allowed lower level tolerance: {}".format(
            self._get_lower_tolerance(jac, f)))
        q, r = linalg.qr(jac)
        x_new = x - linalg.solve(r.T @ r, r.T) @ q.T @ f
        self._save('x', x_new)
        if not self.abort(residual):
            logger.info("New x value:\n{}\nNew jac value:\n{}".format(
                x_new, jac))
            return x_new, True
        else:
            return x_new, False

    def _get_jacobian(self, x):
        if self._jac is None:
            return self._nllq.objective.get_jac(x)
        else:
            return self._jac(x)

    def _get_x0(self, x0):
        if x0 is None:
            x0 = self._variables.current_values

        self._save('x', x0)
        return x0

    def _run(self, x0=None):
        self._save('atol', self.abs_tol)
        x = self._get_x0(x0)
        for i in range(self.max_iter):
            self._save('iteration', i)
            logger.info("Starting iteration: {}".format(i))
            f = self._nllq.objective.residual(x)
            x, cont = self._newton_step(x, f)
            if not cont:
                break
            self._variables.current_values = x
        solver_params = {'iter': i, 'intermediates': self._intermediates}
        return Solution(x, solver_params)


solver_container_factory.register_solver(LocalNonLinearLeastSquares,
                                         GaussNewton,
                                         default=True)
