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

from ...multilevel_object import local_object_factory
from ...variables import NullVariables
from . import Objective
from .lqr import LQRObjective


class ParameterLQRObjective(Objective):
    def __init__(self, higher_level_variables, local_level_variables, time, q,
                 s, r, final_weight):
        lower_level_variables = NullVariables()
        super().__init__(lower_level_variables, higher_level_variables,
                         local_level_variables)
        self._q = q
        self._s = s
        self._r = r
        self._m = final_weight
        self._final_weight = final_weight
        self._time = time

    @property
    def q(self):
        return self._q

    @property
    def s(self):
        return self._s

    @property
    def r(self):
        return self._r

    @property
    def final_weight(self):
        return self._m

    @property
    def time(self):
        return self._time

    def integral_weights(self, *args, **kwargs):
        self._current_q = self._q(*args, **kwargs)
        self._current_r = self._r(*args, **kwargs)
        self._current_s = self._s(*args, **kwargs)
        q = self._current_q
        r = self._current_r
        s = self._current_s
        weights = jnp.block([[q, s], [s.T.conj(), r]])
        return weights


class AutomaticLocalLQRObjective(LQRObjective):
    def __init__(self, global_object):
        self._global_object = global_object
        time = global_object.time
        q, s, r, m = (global_object.localize_method(method)
                      for method in (global_object.q, global_object.s,
                                     global_object.r,
                                     global_object.final_weight))

        super().__init__(time, q, s, r, m)


local_object_factory.register_localizer(ParameterLQRObjective,
                                        AutomaticLocalLQRObjective)
