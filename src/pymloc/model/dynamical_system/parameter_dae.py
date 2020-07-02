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
from typing import Callable
from typing import Optional
from typing import Tuple

import jax
import jax.numpy as np

from ...types import ParameterTimeCallable
from ..multilevel_object import MultiLevelObject
from ..multilevel_object import local_object_factory
from ..variables import VariablesContainer
from .representations import LinearFlowRepresentation

logger = logging.getLogger(__name__)

TimeParameters = Tuple[float, np.ndarray]
ParameterTime = Tuple[np.ndarray, float]


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
                 n: int,
                 residual: Optional[Callable] = None):
        super().__init__(lower_level_variables, higher_level_variables,
                         local_level_variables)
        if residual is not None:
            self.residual = residual
        self.nn: int = n


class LinearParameterDAE(ParameterDAE):
    r"""Class for parameter dependent linear differential algebraic equations of the form

.. math::
    E(t, \theta)\dot{x} = A(t, \theta)x + f(t, \theta)

    or (ommiting time and parameter arguments)

.. math::
    E(\frac{\mathrm d}{\mathrm dt}E^+E{x}) = Ax + f.


    All coefficients are assumed sufficiently smooth.
    The system is assumed to be strangeness-free.
    All quantities according to the definitions in Kunkel, Mehrmann (2006) for every fixed parameter value.
    """
    def __init__(self,
                 ll_vars,
                 hl_vars,
                 loc_vars,
                 e: ParameterTimeCallable,
                 a: ParameterTimeCallable,
                 f: ParameterTimeCallable,
                 n: int,
                 der_e: Optional[ParameterTimeCallable] = None):
        self._e = e
        self._a = a
        self._f = f
        self._der_e = der_e
        super().__init__(ll_vars, hl_vars, loc_vars, n, self.residual)

    @property
    def e(self) -> ParameterTimeCallable:
        return self._e

    @property
    def a(self) -> ParameterTimeCallable:
        return self._a

    @property
    def f(self) -> ParameterTimeCallable:
        return self._f

    @property
    def der_e(self) -> Optional[ParameterTimeCallable]:
        return self._der_e

    @property
    def e_theta(self) -> ParameterTimeCallable:
        return jac_jax_reshaped(self.e, (self.nn, self.nn))

    @property
    def a_theta(self) -> ParameterTimeCallable:
        return jac_jax_reshaped(self.a, (self.nn, self.nn))

    @property
    def f_theta(self) -> ParameterTimeCallable:
        return jac_jax_reshaped(self.f, (self.nn, ))

    def e_plus(self, t: float, param: np.ndarray) -> np.ndarray:
        return np.linalg.pinv(self.e(t, param))

    def projectors(self, t: float,
                   param: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        eplus = self.e_plus(t, param)
        e = self.e(t, param)
        p_z = eplus @ e
        p_check_z = e @ eplus
        return p_z, p_check_z

    def p_z(self, t: float, param: np.ndarray) -> np.ndarray:
        return self.projectors(t, param)[0]

    def _d_a_part(self, t: float, param: np.ndarray) -> np.ndarray:
        p_z, p_check_z = self.projectors(t, param)
        a = self.a(t, param)
        identity = np.eye(self.nn)
        da_part = (identity - p_check_z) @ a @ (identity - p_z)
        return np.linalg.pinv(da_part)

    def d_a(self, t: float, param: np.ndarray) -> np.ndarray:
        a = self.a(t, param)
        return self._d_a_part(t, param) @ a

    def f_a(self, t: float, param: np.ndarray) -> np.ndarray:
        f = self.f(t, param)
        return self._d_a_part(t, param) @ f

    def projector_cal(self, t: float, param: np.ndarray) -> np.ndarray:
        p_z = self.p_z(t, param)
        d_a = self.d_a(t, param)
        ident = np.identity(self.nn)
        return (ident - d_a) @ p_z

    def residual(self, hl_vars: VariablesContainer,
                 loc_vars: VariablesContainer, ll_vars: VariablesContainer):
        p, = hl_vars.current_values
        xdot, x, t = loc_vars.current_values
        e = self.e(p, t)
        a = self.a(p, t)
        return e @ xdot - a @ x


# TODO: Automatic generation should take into consideration appropriate representations. For now LinearFlowRepresentation is general enough.
class AutomaticLinearDAE(LinearFlowRepresentation):
    def __init__(self, parameter_dae: LinearParameterDAE):
        self._parameter_dae = parameter_dae
        variables = parameter_dae.local_level_variables
        nn = parameter_dae.nn
        e, a, f, der_e = (parameter_dae.localize_method(method)
                          for method in (parameter_dae.e, parameter_dae.a,
                                         parameter_dae.f, parameter_dae.der_e))
        super().__init__(variables, e, a, f, nn, der_e)


local_object_factory.register_localizer(LinearParameterDAE, AutomaticLinearDAE)
