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

import jax
import jax.numpy as np

from ..multilevel_object import MultiLevelObject
from ..multilevel_object import local_object_factory
from .representations import LinearFlowRepresentation

logger = logging.getLogger(__name__)


# TODO: Seems to be fixed in recent versions of jax. Use library version!
def jac_jax_reshaped(fun, shape, *args, **kwargs):
    def fun_raveled(*args_fun, **kwargs_fun):
        return fun(*args_fun, **kwargs_fun).ravel()

    def jac_reshaped(*points, **kw_points):
        jac = jax.jacobian(fun_raveled, *args, **kwargs)
        jac_eval = np.atleast_2d(jac(*points, **kw_points).T).T
        diff_dim = jac_eval.shape[1]
        logger.debug("Reshaping jacobian with original shape: {}".format(
            jac_eval.shape))
        jac_eval_tmp = jac_eval.ravel(order='F').reshape(diff_dim, *shape)
        return np.einsum('i...->...i',
                         jac_eval_tmp)  # Transpose for column major

    return jac_reshaped


class ParameterDAE(MultiLevelObject):
    def __init__(self,
                 lower_level_variables,
                 higher_level_variables,
                 local_level_variables,
                 n,
                 residual=None):
        super().__init__(lower_level_variables, higher_level_variables,
                         local_level_variables)
        if residual is not None:
            self.residual = residual
        self.nn = n


class LinearParameterDAE(ParameterDAE):
    def __init__(self, ll_vars, hl_vars, loc_vars, e, a, f, n, der_e=None):
        self._e = e
        self._a = a
        self._f = f
        self._der_e = der_e
        super().__init__(ll_vars, hl_vars, loc_vars, n, self.residual)

    @property
    def e(self):
        return self._e

    @property
    def a(self):
        return self._a

    @property
    def f(self):
        return self._f

    @property
    def der_e(self):
        return self._der_e

    @property
    def e_theta(self):
        return jac_jax_reshaped(self.e, (self.nn, self.nn))

    @property
    def a_theta(self):
        return jac_jax_reshaped(self.a, (self.nn, self.nn))

    @property
    def f_theta(self):
        return jac_jax_reshaped(self.f, (self.nn, ))

    def e_plus(self, *args):
        return np.linalg.pinv(self.e(*args))

    def projectors(self, *args):
        eplus = self.e_plus(*args)
        e = self.e(*args)
        p_z = eplus @ e
        p_check_z = e @ eplus
        return p_z, p_check_z

    def p_z(self, *args):
        return self.projectors(*args)[0]

    def _d_a_part(self, *args):
        p_z, p_check_z = self.projectors(*args)
        a = self.a(*args)
        identity = np.eye(self.nn)
        da_part = (identity - p_check_z) @ a @ (identity - p_z)
        return np.linalg.pinv(da_part)

    def d_a(self, *args):
        a = self.a(*args)
        return self._d_a_part(*args) @ a

    def f_a(self, *args):
        f = self.f(*args)
        return self._d_a_part(*args) @ f

    def projector_cal(self, *args):
        p_z = self.p_z(*args)
        d_a = self.d_a(*args)
        ident = np.identity(self.nn)
        return (ident - d_a) @ p_z

    def residual(self, hl_vars, loc_vars, ll_vars):
        p, = hl_vars.current_value
        xdot, x, t = loc_vars.current_value
        e = self.e(p, x, t)
        a = self.a(p, x, t)
        return e @ xdot - a @ x


# TODO: Automatic generation should take into consideration appropriate representations. For now LinearFlowRepresentation is general enough.
class AutomaticLinearDAE(LinearFlowRepresentation):
    def __init__(self, parameter_dae):
        self._parameter_dae = parameter_dae
        variables = parameter_dae.local_level_variables
        nn = parameter_dae.nn
        e, a, f, der_e = (parameter_dae.localize_method(method)
                          for method in (parameter_dae.e, parameter_dae.a,
                                         parameter_dae.f, parameter_dae.der_e))
        super().__init__(variables, e, a, f, nn, der_e)


local_object_factory.register_localizer(LinearParameterDAE, AutomaticLinearDAE)
