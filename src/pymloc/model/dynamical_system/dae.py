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
from typing import Dict
from typing import Optional

import jax.numpy as jnp
import numpy as np

from ...model.variables.container import StateVariablesContainer
from ...types import TimeCallable

logger = logging.getLogger(__name__)


class DAE:
    """Base class for differential algebraic systems.

    Currently, only the linear case is functional."""
    def __init__(self, variables: StateVariablesContainer, n: int):
        if not isinstance(variables, StateVariablesContainer):
            raise TypeError(variables)
        self._variables = variables
        self._nm = self._variables.n_states
        if n != self._nm and n != self._nm[0]:
            raise ValueError(
                "Number of variables is {}, but has to equal number of equations, which is {}"
                .format(self._nm, n))
        self._nn: int = n
        self._index: Optional[int] = None
        self._current_t: Dict[str, float] = dict()
        self._current_rank: Optional[int] = None

    @property
    def nm(self):
        """
        nm: number of variables
        """
        return self._nm

    @property
    def nn(self):
        """
        nn: number of equations
        """
        return self._nn

    @property
    def index(self):
        """
        index: differentiation index of the DAE.

        Currently only support strangeness-free DAEs, i.e. differentiation index of up to 1.
        """
        return self._index

    @property
    def variables(self):
        """
        variables: Variables instance
        """
        return self._variables


class LinearDAE(DAE):
    r"""Class for linear differential algebraic equations of the form

    .. math::
        E\dot{x} = Ax + f

    or

    .. math::
        E\frac{\mathrm d}{\mathrm dt}(E^+E{x}) = Ax + f.


    All coefficients are assumed sufficiently smooth. The system is assumed to be strangeness-free.
    All quantities according to the definitions in Kunkel, Mehrmann (2006).
    """
    def __init__(self,
                 variables: StateVariablesContainer,
                 e: TimeCallable,
                 a: TimeCallable,
                 f: TimeCallable,
                 n: int,
                 der_e: Optional[TimeCallable] = None,
                 constant_coefficients: bool = True):
        self._constant_coefficients: bool = constant_coefficients
        """bla"""
        super().__init__(variables, n)
        self._e = e
        self._a = a
        self._f = f
        self._rank = None
        self.reset()
        # TODO: Make more general!
        self.init_rank()
        if der_e is None:
            der_e = self._der_e_numerical
        self._der_e = der_e
        self._current_ahat = np.zeros((n, n), order='F')
        self._current_ehat = np.zeros((n, n), order='F')
        var_shape = self._variables.n_states
        self._current_fhat = np.zeros(var_shape, order='F')

    def reset(self) -> None:
        """Resets all DAE objects. Removes stored current values."""
        logger.debug("Current a:\n{}\n{}".format(self._a, self._a(0.)))
        self._current_t = dict()

    @property
    def constant_coefficients(self) -> bool:
        """
        Describes, whether we have a constant (in time) coefficients system.

        If True, many computations are only necessary once.

        :type: bool
        """
        return self._constant_coefficients

    @constant_coefficients.setter
    def constant_coefficients(self, value):
        self._constant_coefficients = value

    @property
    def rank(self) -> int:
        """
        Describes the rank of the matrix E and is assumed to be constant.

        :type: int
        """
        if self._rank is None:
            raise ValueError("Rank has to be initialized first.")
        return self._rank

    def init_rank(self) -> None:
        """Computes rank initially."""
        # TODO: Choose meaningful timepoint
        self._compute_rank(0.)

    def _compute_rank(self, t: float):
        e = self.e(t)
        rank = jnp.linalg.matrix_rank(e)
        if self._current_rank is not None and rank != self._current_rank:
            raise ValueError(
                "Rank change in parameters detected. Not supported and may lead to wrong results."
            )
        self._rank = rank

    def der_e(self, t: float) -> np.ndarray:
        r"""Returns the derivative :math:`\dot{E}(t)`."""
        return self._der_e(t)

    def _der_e_numerical(self, t: float) -> np.ndarray:
        # Use tools like jax
        h = 10e-5
        e_h = self.e(t + h)
        e = self.e(t)
        return (e_h - e) / h

    def _check_current_time(self,
                            t: float,
                            method: str,
                            time_varying: bool = False) -> bool:
        if self._current_t.get(method) is None or (
            (time_varying or not self.constant_coefficients)
                and self._current_t[method] != t):
            self._current_t[method] = t
            return True
        else:
            return False

    def e(self, t: float) -> np.ndarray:
        r"""Computes :math:`E(t)`."""
        self._recompute_coefficients(t)
        return self._current_e

    def a(self, t: float) -> np.ndarray:
        r"""Computes :math:`A(t)`."""
        self._recompute_coefficients(t)
        return self._current_a

    def f(self, t: float) -> np.ndarray:
        r"""Computes :math:`f(t)`."""
        self._recompute_inhomogeinity(t)
        return self._current_f

    def eplus(self, t: float) -> np.ndarray:
        r"""Computes the pseudo inverse :math:`E^+(t)`."""
        self._recompute_quantities(t)
        return self._current_eplus

    def t2(self, t: float) -> np.ndarray:
        r"""Computes the selector matrix :math:`T_2(t)`."""
        self._recompute_quantities(t)
        return self._current_ttprime_h[:, :self.rank]

    def t2prime(self, t: float) -> np.ndarray:
        r"""Computes the selector matrix :math:`T_2^{\prime}(t)`."""
        self._recompute_quantities(t)
        return self._current_ttprime_h[:, self.rank:]

    def z1(self, t: float) -> np.ndarray:
        r"""Computes the selector matrix :math:`Z_1(t)`."""
        self._recompute_quantities(t)
        return self._current_zzprime[:, :self.rank]

    def z1prime(self, t: float) -> np.ndarray:
        r"""Computes the selector matrix :math:`Z_1^{\prime}(t)`."""
        self._recompute_quantities(t)
        return self._current_zzprime[:, self.rank:]

    def ehat_1(self, t: float) -> np.ndarray:
        r"""Computes the reduced system matrix :math:`\hat E_1(t)`."""
        self._recompute_quantities(t)
        return self._current_ehat[:self.rank, :]

    def ehat_2(self, t: float) -> np.ndarray:
        r"""Computes the reduced system matrix :math:`\hat E_2(t)`."""
        self._recompute_quantities(t)
        return self._current_ehat[self.rank:, :]

    def ahat_1(self, t: float) -> np.ndarray:
        r"""Computes the reduced system matrix :math:`\hat A_1(t)`."""
        self._recompute_quantities(t)
        return self._current_ahat[:self.rank, :]

    def ahat_2(self, t: float) -> np.ndarray:
        r"""Computes the reduced system matrix :math:`\hat A_2(t)`."""
        self._recompute_quantities(t)
        return self._current_ahat[self.rank:, :]

    def fhat_1(self, t: float) -> np.ndarray:
        r"""Computes the reduced system matrix :math:`\hat{f}_1(t)`."""
        self._recompute_fhat(t)
        return self._current_fhat[:self.rank, ...]

    def fhat_2(self, t: float) -> np.ndarray:
        r"""Computes the reduced system matrix :math:`\hat{f}_2(t)`."""
        self._recompute_fhat(t)
        return self._current_fhat[self.rank:, ...]

    def _recompute_coefficients(self, t: float) -> None:
        if self._check_current_time(t, "coefficients"):
            e = self._e(t)
            a = self._a(t)
            self._current_e = e
            self._current_a = a

    def _recompute_inhomogeinity(self, t: float) -> None:
        if self._check_current_time(t, "inhomogeinity", time_varying=True):
            f = self._f(t)
            self._current_f = f

    def _recompute_quantities(self, t: float) -> None:
        if self._check_current_time(t, "quantities"):
            e = self.e(t)
            a = self.a(t)
            zzprime, sigma, ttprime_h = jnp.linalg.svd(e)
            rank = self.rank
            self._current_ttprime_h = ttprime_h.T
            self._current_zzprime = zzprime
            self._current_eplus = self.t2(t) @ np.linalg.solve(
                np.diag(sigma[:rank]),
                self.z1(t).T)
            ehat_1 = self.z1(t).T @ e
            self._current_ehat[:rank, :] = ehat_1
            ahat_1 = self.z1(t).T @ a
            ahat_2 = self.z1prime(t).T @ a
            self._current_ahat[:rank, :] = ahat_1
            self._current_ahat[rank:, :] = ahat_2

    def _recompute_fhat(self, t: float) -> None:
        self._recompute_quantities(t)
        if self._check_current_time(t, "fhat", time_varying=True):
            rank = self.rank
            f = self.f(t)
            fhat_1 = self.z1(t).T @ f
            fhat_2 = self.z1prime(t).T @ f
            self._current_fhat[:rank, ...] = fhat_1
            self._current_fhat[rank:, ...] = fhat_2
