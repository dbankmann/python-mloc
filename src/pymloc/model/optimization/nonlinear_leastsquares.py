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
from ..multilevel_object import local_object_factory
from ..solvable import Solvable
from .constraints.constraint import NullConstraint
from .optimization import AutomaticLocalOptimizationObject
from .optimization import OptimizationObject


class LocalNonLinearLeastSquares(Solvable):
    pass


class NonLinearLeastSquares(OptimizationObject):
    r"""Implementation of a nonlinear least squares problem of the form

    .. math::
    {\sum_{k=1}^q{\|f(t_k, \theta)-\xi_k\|^2}}

    for given data points :math:`(t_k, \xi_k)` and the unknown :math:`\theta \in \R^p`.
    """
    _local_object_class = LocalNonLinearLeastSquares

    def __init__(self, objective_obj, ll_vars, hl_vars, loc_vars):
        vars = (ll_vars, hl_vars, loc_vars)
        super().__init__(objective_obj, NullConstraint(*vars), *vars)


class AutomaticLocalNonLinearLeastSquares(AutomaticLocalOptimizationObject,
                                          LocalNonLinearLeastSquares):
    ...


local_object_factory.register_localizer(NonLinearLeastSquares,
                                        AutomaticLocalNonLinearLeastSquares)
