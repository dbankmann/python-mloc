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
import numpy as np

from ....misc import unstack
from ...multilevel_object import local_object_factory
from .objective import AutomaticLocalObjective
from .objective import Objective


class NonLinearLeastSquares(Objective):
    def __init__(self, lower_level_variables, higher_level_variables,
                 local_level_variables, rhs):
        super().__init__(lower_level_variables, higher_level_variables,
                         local_level_variables)
        self._rhs = rhs

    def residual(self, ll_vars, hl_vars, loc_vars):
        return self._rhs(ll_vars, hl_vars, loc_vars)

    def get_jac(self, ll_vars, hl_vars, loc_vars):
        loc_sens = self._get_loc_sens()
        solver_args = self._get_ll_solver_args()
        ll_sens_sol = self.lower_level_variables.get_sensitivities(
            **solver_args)
        ll_sens = unstack(ll_sens_sol.solution)
        return ll_sens + loc_sens

    def _get_loc_sens(self):
        # TODO:Implement
        return np.zeros(1)

    def _get_ll_solver_args(self):
        kwargs = dict()
        try:
            si = self.local_level_variables.associated_problem.solver_instance
            abst = si.lower_abs_tolerance
            kwargs['abs_tol'] = abst
            kwargs['rel_tol'] = si.rel_tol
        except AttributeError:
            pass
        return kwargs


class AutomaticLocalNonLinearLeastSquares(AutomaticLocalObjective):
    def __init__(self, global_objective):
        super().__init__(global_objective)
        self.get_jac = global_objective.localize_method(
            global_objective.get_jac)


local_object_factory.register_localizer(NonLinearLeastSquares,
                                        AutomaticLocalNonLinearLeastSquares)
