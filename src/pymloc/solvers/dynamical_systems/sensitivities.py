import logging
from abc import ABC
from abc import abstractmethod
from typing import Type

import jax
import numpy as np

from ...model.dynamical_system.boundary_value_problem import BoundaryValueProblem
from ...model.dynamical_system.representations import LinearFlowRepresentation
from ...model.sensitivities.boundary_dae import BVPSensitivities
from ...model.variables.container import StateVariablesContainer
from ..base_solver import BaseSolver
from ..base_solver import TimeSolution

logger = logging.getLogger(__name__)


class SensitivitiesInhomogeneity(ABC):
    """Baseclass for the inhomogeneity used in both -- forward and adjoint --
    sensitivity computations"""
    @abstractmethod
    def __init__(self, sensitivities: SensitivitiesSolver,
                 localized_bvp: BoundaryValueProblem, solution: TimeSolution,
                 parameter: np.ndarray):
        self._sensitivities = sensitivities
        self._dynamical_system = self._sensitivities.dynamical_system
        self._parameter = parameter
        self._solution = solution
        self._localized_bvp = localized_bvp
        self._bvp_param = self._sensitivities.bvp_param
        self._time_interval = self._localized_bvp.time_interval
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
        raise NotImplementedError  # TODO: Implement

    @abstractmethod
    def capital_f_theta(self, t):
        pass

    @abstractmethod
    def _complement_f_tilde(self, t):
        pass

    def capital_f_tilde(self, t):
        return self._complement_f_tilde(t) - self.capital_f_theta(t)
        return self.capital_f_theta(t) - self._complement_f_tilde(t)

    def get_capital_fs(self):
        return self.capital_f_theta, self.capital_f_tilde, self.eplus_e_theta

    def _set_eplus_e_derivatives(self, parameter):
        """Sets several derivative functions wrt. parameters"""
        dae = self._dynamical_system

        epe = dae.p_z
        epe_theta = jax.jacobian(epe)
        epe_theta_t = jax.jacobian(epe_theta, argnums=1)
        parameter = np.atleast_1d(parameter)

        def eplus_e_theta(t):
            "Computes the derivative dp(E^+E)"
            return epe_theta(parameter, t)

        def eplus_e_theta_t(t):
            "Computes the derivative dpdt(E^+E)"
            return epe_theta_t(parameter, t)

        def e_ddt_epluse_theta(t):
            "Computes the quantity E @ dpdt(E^+E)"
            e = dae.e(parameter, t)
            return np.einsum('ij, jkp->ikp', e, eplus_e_theta_t(t))

        self.eplus_e_theta = eplus_e_theta
        self.eplus_e_theta_t = eplus_e_theta_t
        self.e_ddt_epluse_theta = e_ddt_epluse_theta


class SensInhomWithTimeDerivative(SensitivitiesInhomogeneity):
    "Subclass for case a) described in SensitivitiesSolver class"

    def capital_f_theta(self, t: np.float):
        f_tilde = np.einsum(
            'ijk,j->ik', self.a_dif(t), self._solution(t)) - np.einsum(
                'ijk,j->ik', self.e_dif(t), self.x_dot(t)) + self.f_dif(t)
        return f_tilde

    def capital_f_tilde(self, t: np.float):
        return self.capital_f_theta(t)

    def _complement_f_tilde(self, t: np.float):
        return np.zeros(0)


class SensInhomProjection(SensitivitiesInhomogeneity):
    "Subclass for case b) described in SensitivitiesSolver class"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.capital_f_theta(
            self._time_interval.t_0)  # For checking subset. #TODO: Refactor

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
    "Subclass for case c) described in SensitivitiesSolver class"

    def capital_f_theta(self, t):
        f_tilde = -np.einsum('ijk,j->ik', self.e_dif(t),
                             self.x_d_dot(t)) + np.einsum(
                                 'ijk,j->ik',
                                 self.a_dif(t) + self.e_ddt_epluse_theta(t),
                                 self._solution(t)) + self.f_dif(t)
        return f_tilde

    def _complement_f_tilde(self, t):
        a = self._dynamical_system.a(self._parameter, t)
        epep = self.eplus_e_theta(t)
        compl = np.einsum('ij, jkp,k->ip', a, epep, self._solution(t))
        return compl

    def summand_2(self, tau):
        selector = self._bvp_param.selector(self._parameter)
        dyn_param = self._bvp_param.dynamical_system
        proj_cal_theta = self.projector_cal_theta(tau)
        if self._time_interval.at_bound(tau):
            sol = self._solution(tau)
        else:
            sol = self.x_d(tau)
        temp = np.einsum('ijp,j...->i...p', proj_cal_theta, sol)
        return selector @ temp

    def f_a_theta(self, tau):
        dyn_param = self._bvp_param.dynamical_system
        fa_theta = jax.jacobian(dyn_param.f_a)
        fa_th_eval = fa_theta(self._parameter, tau)
        if fa_th_eval.ndim == 1:  # TODO: Homogenize. Better always use arrays; also for float inputs
            fa_th_eval = fa_th_eval[..., np.newaxis]

        return fa_th_eval

    def projector_cal_theta(self, tau):
        dyn_param = self._bvp_param.dynamical_system
        projector_cal = dyn_param.projector_cal
        projector_cal_theta = jax.jacobian(projector_cal)
        projector_cal_theta_eval = projector_cal_theta(self._parameter, tau)
        if projector_cal_theta_eval.ndim == 2:  # TODO: Homogenize. Better always use arrays; also for float inputs
            projector_cal_theta_eval = projector_cal_theta_eval[...,
                                                                np.newaxis]
        return projector_cal_theta_eval


class SensitivitiesSolver(BaseSolver, ABC):
    """Baseclass for both Sensitivity solvers."""
    capital_f_default_class: Type[SensitivitiesInhomogeneity]
    _capital_f_classes = (SensInhomWithTimeDerivative, SensInhomProjection,
                          SensInhomProjectionNoSubset)

    @abstractmethod
    def __init__(self, bvp_param: BVPSensitivities, *args, **kwargs):
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
        The adjoint approach can be used in all cases.
        However, in the most general case c) additional derivatives of the original data are necessary
        Approach a) has the additional disadvantage that the product E_p @ x_dot may not be available directly from most DAE solvers.
        """
        fs_instance = self.capital_f_class(self, *args, **kwargs)
        self._capital_fs_instance = fs_instance
        return fs_instance.get_capital_fs()
