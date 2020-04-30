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
from abc import ABC
from abc import abstractmethod
from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
import scipy
import scipy.linalg as linalg

from pymloc.model.dynamical_system.flow_problem import LinearFlow

from ...model.dynamical_system.boundary_value_problem import BoundaryValueProblem
from ...model.dynamical_system.boundary_value_problem import BoundaryValues
from ...model.dynamical_system.boundary_value_problem import MultipleBoundaryValueProblem
from ...model.dynamical_system.boundary_value_problem import MultipleBoundaryValues
from ...model.dynamical_system.initial_value_problem import InitialValueProblem
from ...model.dynamical_system.representations import LinearFlowRepresentation
from ...model.sensitivities.boundary_dae import BVPSensitivities
from ...model.variables.container import StateVariablesContainer
from ...model.variables.time_function import Time
from ...solver_container import solver_container_factory
from ..base_solver import BaseSolver
from ..base_solver import TimeSolution

logger = logging.getLogger(__name__)


class SensitivitiesInhomogeneity(ABC):
    def __init__(self, sensitivities, localized_bvp, solution, parameter):
        self._sensitivities = sensitivities
        self._dynamical_system = self._sensitivities.dynamical_system
        self._parameter = parameter
        self._solution = solution
        self._localized_bvp = localized_bvp
        self._bvp_param = self._sensitivities.bvp_param
        self._time_interval = self._localized_bvp.time_interval

        eplus = self._localized_bvp.dynamical_system.eplus
        self._set_eplus_e_derivatives(self._parameter)

    @property
    def solution(self):
        return self._solution

    def a_dif(self, t):
        return self._dynamical_system.a_theta(self._parameter, t)

    def e_dif(self, t):
        return self._dynamical_system.e_theta(self._parameter, t)

    def f_dif(self, t):
        return self._dynamical_system.f_theta(self._parameter, t)

    def x_d(self, t):
        return self._localized_bvp.dynamical_system.x_d(t, self._solution(t))

    def x_d_dot(self, t):
        fd = self._localized_bvp.dynamical_system.f_d(t)
        ddxd = np.einsum('ij,j->i',
                         self._localized_bvp.dynamical_system.d_d(t),
                         self.x_d(t))
        return ddxd + fd

    def x_dot(self, t):
        raise NotImplementedError  #TODO: Implement

    @abstractmethod
    def capital_f_theta(self, t):
        pass

    @abstractmethod
    def _complement_f_tilde(self, t):
        pass

    def capital_f_tilde(self, t):
        return self._complement_f_tilde(t) - self.capital_f_theta(t)

    def get_capital_fs(self):
        return self.capital_f_theta, self.capital_f_tilde, self.eplus_e_theta

    def _set_eplus_e_derivatives(self, parameter):
        dae = self._dynamical_system

        epe = dae.p_z
        epe_theta = jax.jacobian(epe)
        epe_theta_t = jax.jacobian(epe_theta, argnums=1)
        parameter = np.atleast_1d(parameter)

        def eplus_e_theta(t):
            return epe_theta(parameter, t)

        def eplus_e_theta_t(t):
            return epe_theta_t(parameter, t)

        def e_ddt_epluse_theta(t):
            e = dae.e(parameter, t)
            return np.einsum('ij, jkp->ikp', e, eplus_e_theta_t(t))

        self.eplus_e_theta = eplus_e_theta
        self.eplus_e_theta_t = eplus_e_theta_t
        self.e_ddt_epluse_theta = e_ddt_epluse_theta


class SensInhomWithTimeDerivative(SensitivitiesInhomogeneity):
    def capital_f_theta(self, t):
        f_tilde = np.einsum(
            'ijk,j->ik', self.a_dif(t), self._solution(t)) - np.einsum(
                'ijk,j->ik', self.e_dif(t), self.x_dot(t)) + self.f_dif(t)
        return f_tilde

    def capital_f_tilde(self, t):
        return self.capital_f_theta(t)

    def _complement_f_tilde(self, t):
        return np.zeros(0)


class SensInhomProjection(SensitivitiesInhomogeneity):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.capital_f_theta(
            self._time_interval.t_0)  #To check subset. #TODO: Refactor

    def _check_subset(self, epe_theta, p_z):
        epe_theta_pz = np.einsum('ijk, jl-> ilk', epe_theta, p_z)
        if not np.allclose(epe_theta, epe_theta_pz):
            raise ValueError(
                "Approach not appropriate for systems, where (E^+E)_theta (E^+E) != (E^+E)_theta!"
            )

    def capital_f_theta(self, t):
        dae = self._dynamical_system
        f_tilde = np.einsum('ijk,j->ik',
                            self.a_dif(t) + self.e_ddt_epluse_theta(t),
                            self._solution(t)) - np.einsum(
                                'ijk,j->ik', self.e_dif(t),
                                self.x_d_dot(t)) + self.f_dif(t)
        e = self._dynamical_system.e(self._parameter, t)
        p_z = dae.p_z(self._parameter, t)
        e_pe_p_eval = self.eplus_e_theta(t)
        self._check_subset(e_pe_p_eval, p_z)
        tmp1 = np.einsum('ij, jkl, k ->il', e, e_pe_p_eval, self.x_d_dot(t))
        tmp2 = np.einsum('ij, jkl, k ->il', e, self.e_ddt_epluse_theta(t),
                         self.x_d(t))
        return f_tilde - tmp1 - tmp2

    def _complement_f_tilde(self, t):
        return np.zeros(1)

    def _get_f_tilde_dae(self, localized_bvp, f_tilde):
        n = self._dynamical_system.nn
        nparam = self._bvp_param.n_param
        shape = (n, nparam)
        variables = StateVariablesContainer(shape)
        dyn_sys = localized_bvp.dynamical_system
        return LinearFlowRepresentation(variables, dyn_sys.e, dyn_sys.a,
                                        f_tilde, n)

    def temp2_f_a_theta(self, capital_f_theta, tau):

        selector = self._bvp_param.selector(self._parameter)
        da = self._localized_bvp.dynamical_system.d_a(tau)
        f_theta = self._bvp_param.dynamical_system.f_theta(
            self._parameter, tau)
        temp_bvp = self._get_f_tilde_dae(self._localized_bvp, capital_f_theta)
        temp12 = selector @ temp_bvp.f_a(tau)
        temp2 = selector @ da @ f_theta
        return temp2 + temp12


class SensInhomProjectionNoSubset(SensInhomProjection):
    def a_times_epluse_theta(self, t):
        a_times_epluse_p = np.einsum(
            'ij,jkp->ikp', self._dynamical_system.a(self._parameter, t),
            self.eplus_e_theta(t))
        return a_times_epluse_p

    def capital_f_theta(self, t):
        f_tilde = -np.einsum('ijk,j->ik', self.e_dif(t),
                             self.x_d_dot(t)) + np.einsum(
                                 'ijk,j->ik',
                                 self.a_dif(t) + self.e_ddt_epluse_theta(t),
                                 self._solution(t)) + self.f_dif(t)
        return f_tilde

    def _complement_f_tilde(self, t):
        atepep = self.a_times_epluse_theta(t)
        a = self._dynamical_system.a(self._parameter, t)
        epep = self.eplus_e_theta(t)
        compl = np.einsum('ij, jkp,k->ip', a, epep, self._solution(t))
        return compl

    def temp2_f_a_theta(self, capital_f_theta, tau):
        selector = self._bvp_param.selector(self._parameter)
        da = self._localized_bvp.dynamical_system.d_a(tau)
        dyn_param = self._bvp_param.dynamical_system
        da_theta = jax.jacobian(dyn_param.d_a)
        fa_theta = jax.jacobian(dyn_param.f_a)
        temp = np.einsum('ijp,j...->i...p', self.eplus_e_theta(tau),
                         self.solution(tau))
        temp = temp - np.einsum('ij,j...p->i...p', da, temp)
        da_th_eval = da_theta(self._parameter, tau)
        fa_th_eval = fa_theta(self._parameter, tau)
        if da_th_eval.ndim == 2:  #TODO: Homogenize. Better always use arrays; also for float inputs
            da_th_eval = da_th_eval[..., np.newaxis]
            fa_th_eval = fa_th_eval[..., np.newaxis]
        temp2 = np.einsum('ijp, j...->i...p', da_th_eval, self.x_d(tau))
        val = temp - temp2 - fa_th_eval
        return selector @ val


class SensitivitiesSolver(BaseSolver):
    _capital_f_classes = (SensInhomWithTimeDerivative, SensInhomProjection,
                          SensInhomProjectionNoSubset)

    def __init__(self, bvp_param, *args, **kwargs):
        if not isinstance(bvp_param, BVPSensitivities):
            raise TypeError(bvp_param)
        self._bvp_param = bvp_param
        self._dynamical_system = bvp_param.dynamical_system
        self._nn = self._dynamical_system.nn
        self._time_interval = self._bvp_param.time_interval
        self._boundary_values = self._bvp_param.boundary_value_problem.boundary_values
        self._der_eplus_e_theta = None
        self._der_eplus_e_theta_is_zero = False
        self.capital_f_class = self.capital_f_default_class
        super().__init__(*args, **kwargs)

    @property
    def bvp_param(self):
        return self._bvp_param

    @property
    def dynamical_system(self):
        return self._dynamical_system

    @property
    def capital_f_class(self):
        return self._capital_f_class

    @capital_f_class.setter
    def capital_f_class(self, value):
        if value in self._capital_f_classes:
            self._capital_f_class = value
        else:
            raise ValueError(value)

    def _get_capital_fs(self, *args, **kwargs):
        """
        Computes the capital_f_tilde quantity, that is used in the computation of both, forward and adjoint sensitivities.
        The concrete result depends on certain conditions.

        1. Compute forward or adjoint sensitivities?
        2. Regularity level
           a) Time derivative of the algebraic variables exist
           b) dp(E^+E) E^+E = dp(E^+E)
           c) b) is not fulfilled

        The forward approach can only be used in case a) and b).
        The adjoint approach can be used in all cases. However, in the most general case c) additional derivatives of the original data are necessary
        Approach a) has the additional disadvantage that the product E_p @ x_dot may not be available directly from most DAE solvers.
        """
        fs_instance = self.capital_f_class(self, *args, **kwargs)
        self._capital_fs_instance = fs_instance
        return fs_instance.get_capital_fs()
