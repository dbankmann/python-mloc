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

import jax.numpy as jnp
import numpy as np

from ....types import ParameterCallable
from ....types import ParameterTimeCallable
from ...multilevel_object import local_object_factory
from ...variables import NullVariables
from ...variables import Time
from . import Objective
from .lqr import LQRObjective


class ParameterLQRObjective(Objective):
    r"""Parameter dependent version of the LQR objective.
    The objective function is given by

    .. math::
      {
        x(\theta,\tf)^{\Herm}Kx(\theta,\tf) + \int_{t_0}^{\tf}{
            \begin{pmatrix}
                x(\theta,t)\\
                u(\theta,t)
            \end{pmatrix}^{\Herm}
            \begin{bmatrix}
                Q(\theta,t)  & S(\theta,t)\\
                S^{\Herm}(\theta,t)& R(\theta,t)
            \end{bmatrix}
            \begin{pmatrix}
                x(\theta,t)\\
                u(\theta,t)
            \end{pmatrix}
    \mathrm{d}t}}.
    """
    def __init__(self, higher_level_variables, local_level_variables,
                 time: Time, q: ParameterTimeCallable,
                 s: ParameterTimeCallable, r: ParameterTimeCallable,
                 final_weight: ParameterCallable):
        lower_level_variables = NullVariables()
        super().__init__(lower_level_variables, higher_level_variables,
                         local_level_variables)
        self._q = q
        self._s = s
        self._r = r
        self._final_weight: ParameterCallable = final_weight
        self._time = time

    @property
    def q(self) -> ParameterTimeCallable:
        return self._q

    @property
    def s(self) -> ParameterTimeCallable:
        return self._s

    @property
    def r(self) -> ParameterTimeCallable:
        return self._r

    @property
    def final_weight(self) -> ParameterCallable:
        return self._final_weight

    @property
    def time(self) -> Time:
        return self._time

    def integral_weights(self, theta: np.ndarray, t: float) -> np.ndarray:
        self._current_q = self._q(theta, t)
        self._current_r = self._r(theta, t)
        self._current_s = self._s(theta, t)
        q = self._current_q
        r = self._current_r
        s = self._current_s
        weights = jnp.block([[q, s], [s.T.conj(), r]])
        return weights


class AutomaticLocalLQRObjective(LQRObjective):
    def __init__(self, global_object: ParameterLQRObjective):
        self._global_object = global_object
        time = global_object.time
        q, s, r = (global_object.localize_method(method)
                   for method in (global_object.q, global_object.s,
                                  global_object.r))
        m = global_object.localize_method(global_object.final_weight)

        super().__init__(time, q, s, r, m)


local_object_factory.register_localizer(ParameterLQRObjective,
                                        AutomaticLocalLQRObjective)
