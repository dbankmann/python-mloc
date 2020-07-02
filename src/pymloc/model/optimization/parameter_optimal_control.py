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
import jax
import jax.numpy as jnp
import numpy as np

from ..dynamical_system.parameter_bvp import ParameterBoundaryValueProblem
from ..dynamical_system.parameter_bvp import ParameterBoundaryValues
from ..dynamical_system.parameter_dae import LinearParameterDAE
from ..multilevel_object import local_object_factory
from ..sensitivities.boundary_dae import BVPSensitivities
from ..variables import InputStateVariables
from ..variables import NullVariables
from ..variables import ParameterContainer
from ..variables.container import StateVariablesContainer
from .constraints.parameter_lqr import ParameterLQRConstraint
from .objectives.parameter_lqr import ParameterLQRObjective
from .optimal_control import LQOptimalControl
from .optimization import AutomaticLocalOptimizationObject
from .optimization import OptimizationObject


class ParameterDependentOptimalControl(OptimizationObject):
    """Parameter dependent version of the LQR Optimization object.

    Needs parameter dependent objective and constraint."""
    _local_object_class = LQOptimalControl

    def __init__(self, parameters: ParameterContainer,
                 state_input: InputStateVariables,
                 objective: ParameterLQRObjective,
                 constraint: ParameterLQRConstraint):
        if not isinstance(parameters, ParameterContainer):
            raise TypeError(parameters)
        self._parameters = parameters
        if not isinstance(state_input, InputStateVariables):
            raise TypeError(parameters)
        self._state_input = state_input
        lower_level_variables = NullVariables()
        local_level_variables = state_input
        higher_level_variables = parameters
        self.constraint_object = constraint
        self.objective_object = objective
        if not isinstance(objective, ParameterLQRObjective):
            raise TypeError(objective)
        if not isinstance(constraint, ParameterLQRConstraint):
            raise TypeError(constraint)
        self._time = self.constraint_object.control_system.time
        super().__init__(self.objective_object, self.constraint_object,
                         lower_level_variables, higher_level_variables,
                         local_level_variables)

    @property
    def parameters(self) -> np.ndarray:
        return self._parameters

    @property
    def state_input(self) -> InputStateVariables:
        return self._state_input

    @property
    def objective_object(self) -> ParameterLQRObjective:
        return self._objective_object

    @objective_object.setter
    def objective_object(self, value):
        self._objective_object = value

    @property
    def constraint_object(self) -> ParameterLQRConstraint:
        return self._constraint_object

    @constraint_object.setter
    def constraint_object(self, value):
        self._constraint_object = value

    def get_sensitivities(self, **kwargs) -> BVPSensitivities:
        """Returns a sensitivities object for the parameter dependent boundary problem of the
        necessary conditions.
        """
        boundary_value_problem = self.get_bvp()
        n_param = self._parameters.parameters.dimension
        args = (self.ll_sens_selector, self.ll_sens_selector_shape)
        return BVPSensitivities(boundary_value_problem, n_param, *args,
                                **kwargs)

    # TODO: Refactor parameter and non-parameter functions
    def get_bvp(self) -> ParameterBoundaryValueProblem:
        """Returns the parameter dependent boundary value problem of the necessary conditions."""
        nn = self._state_input.states.dimension
        nm = self._state_input.inputs.dimension
        dim = nm + 2 * nn
        n_param = self._parameters.parameters.dimension
        iweights = self.objective_object.integral_weights
        econtr = self.constraint_object.control_system.augmented_dae.e
        acontr = self.constraint_object.control_system.augmented_dae.a
        fcontr = self.constraint_object.control_system.augmented_dae.f

        zeros = jnp.zeros((nn, nn))
        zeros2 = jnp.zeros((nn + nm, nn + nm))
        zeros3 = jnp.zeros((nn + nm, ))
        zeros4 = jnp.zeros((nn + nm, 2 * nn + nm))
        zeros5 = jnp.zeros((nn, dim))
        zeros51 = jnp.zeros((nm, dim))
        zeros6 = jnp.zeros((nn, nm))

        @jax.jit
        def ecal(p: jnp.ndarray, t: float) -> jnp.ndarray:
            e = econtr(p, t)
            ecal = jnp.block([[zeros, e], [-e.T, zeros2]])
            return ecal

        @jax.jit
        def acal(p: jnp.ndarray, t: float) -> jnp.ndarray:
            a = acontr(p, t)
            iweight = iweights(p, t)
            acal = jnp.block([[zeros, a], [a.T, iweight]])
            return acal

        @jax.jit
        def fcal(p: jnp.ndarray, t: float) -> jnp.ndarray:
            fcal = jnp.block([fcontr(p, t), zeros3])
            return fcal

        @jax.jit
        def gamma_0(p: jnp.ndarray) -> jnp.ndarray:
            gamma_0_arr = jnp.block([[zeros, econtr(p, self._time.t_0)],
                                     [zeros4]])
            return gamma_0_arr

        @jax.jit
        def gamma_f(p: jnp.ndarray) -> jnp.ndarray:
            gamma_f_arr = jnp.block([[zeros5],
                                     [
                                         jnp.identity(nn),
                                         self.objective_object.final_weight(p),
                                         zeros6
                                     ], [zeros51]])
            return gamma_f_arr

        free_dae = self.constraint_object.control_system.free_dae

        @jax.jit
        def gamma_rhs(p: jnp.ndarray) -> jnp.ndarray:
            gamma_rhs_arr = jnp.block([
                free_dae.e(p, self._time.t_0)
                @ self.constraint_object.initial_value(p), zeros3
            ])
            return gamma_rhs_arr

        def z_gamma(p: jnp.ndarray) -> jnp.ndarray:
            time = self._time
            t0 = time.t_0
            tf = time.t_f
            free_dae.higher_level_variables.current_values = p
            loc_dae = free_dae.get_localized_object()
            rank = loc_dae.rank
            z_1_x = loc_dae.z1(t0)
            z_1_l = loc_dae.t2(tf)
            zerosn = jnp.zeros((nn, rank))
            zerosm = jnp.zeros((nm, rank))
            z_g = jnp.block([[z_1_x, zerosn], [zerosn, z_1_l],
                             [zerosm, zerosm]])

            return z_g

        variables = StateVariablesContainer(dim)
        bvs = ParameterBoundaryValues(self.lower_level_variables,
                                      self.higher_level_variables,
                                      variables,
                                      gamma_0,
                                      gamma_f,
                                      gamma_rhs,
                                      dim,
                                      n_param,
                                      z_gamma=z_gamma)
        dyn_sys = LinearParameterDAE(self.lower_level_variables,
                                     self.higher_level_variables, variables,
                                     ecal, acal, fcal, dim)
        param_bvp = ParameterBoundaryValueProblem(self.lower_level_variables,
                                                  self.higher_level_variables,
                                                  variables, self._time,
                                                  dyn_sys, bvs)

        return param_bvp


class AutomaticLQOptimalControl(AutomaticLocalOptimizationObject,
                                LQOptimalControl):
    def __init__(self, global_object: ParameterDependentOptimalControl,
                 **kwargs):
        super().__init__(global_object, **kwargs)
        LQOptimalControl._init_local(self)


local_object_factory.register_localizer(ParameterDependentOptimalControl,
                                        AutomaticLQOptimalControl)
