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
from typing import Dict

import numpy as np

from ....misc import unstack
from ...multilevel_object import local_object_factory
from ...variables import VariablesContainer
from .objective import AutomaticLocalObjective
from .objective import Objective


class NonLinearLeastSquares(Objective):
    """Objective function of nonlinear least squares problems"""
    def __init__(self, lower_level_variables, higher_level_variables,
                 local_level_variables, rhs: Callable[[
                     VariablesContainer, VariablesContainer, VariablesContainer
                 ], np.ndarray]):
        super().__init__(lower_level_variables, higher_level_variables,
                         local_level_variables)
        self._rhs = rhs
        """
        Parameters
        ------------

        rhs: residual function
        """
    def residual(self, ll_vars, hl_vars, loc_vars):
        return self._rhs(ll_vars, hl_vars, loc_vars)

    def get_jac(  # TODO: Fix types to use variables type type: ignore
            self,
            ll_vars,
            hl_vars,
            loc_vars  # type: ignore
    ) -> np.ndarray:
        """Helper method, that computes the jacobian of the residual function.

        Possibly by using sensitivity information from lower level optimizations."""
        loc_sens = self._get_loc_sens()
        solver_args = self._get_ll_solver_args()
        ll_sens_sol = self.lower_level_variables.get_sensitivities(
            **solver_args)
        ll_sens = unstack(ll_sens_sol.solution)
        return ll_sens + loc_sens

    def _get_loc_sens(self):
        # TODO:Implement the case, where the residual also depends on local variables
        return np.zeros(1)

    def _get_ll_solver_args(self) -> Dict[str, float]:
        kwargs = dict()
        # TODO: Should actually be implemented with solver interface.
        try:
            si = self.local_level_variables.associated_problem.solver_instance
            abst = si.lower_abs_tolerance  # type: ignore
            kwargs['abs_tol'] = abst
            kwargs['rel_tol'] = si.rel_tol  # type: ignore
        except AttributeError:
            pass
        return kwargs


class AutomaticLocalNonLinearLeastSquares(AutomaticLocalObjective):
    def __init__(self, global_objective: NonLinearLeastSquares):
        super().__init__(global_objective)
        self.get_jac = global_objective.localize_method(
            global_objective.get_jac)  # type: ignore # TODO: fix!


local_object_factory.register_localizer(NonLinearLeastSquares,
                                        AutomaticLocalNonLinearLeastSquares)
