import jax
import jax.numpy as jnp
import numpy as np
import scipy.integrate

from ..dynamical_system.parameter_bvp import ParameterBoundaryValueProblem
from ..dynamical_system.parameter_bvp import ParameterBoundaryValues
from ..dynamical_system.parameter_dae import LinearParameterDAE
from ..dynamical_system.parameter_ivp import ParameterInitialValueProblem
from ..multilevel_object import MultiLevelObject
from ..multilevel_object import local_object_factory
from ..sensitivities.boundary_dae import BVPSensitivities
from ..variables import InputStateVariables
from ..variables import NullVariables
from ..variables import ParameterContainer
from ..variables.container import StateVariablesContainer
from .constraints.parameter_lqr import ParameterLQRConstraint
from .objectives import Objective
from .objectives.parameter_lqr import ParameterLQRObjective
from .optimal_control import LQOptimalControl
from .optimization import AutomaticLocalOptimizationObject
from .optimization import OptimizationObject


class ParameterDependentOptimalControl(OptimizationObject):
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
        self._constraint = constraint
        self._objective = objective
        if not isinstance(objective, ParameterLQRObjective):
            raise TypeError(objective)
        if not isinstance(constraint, ParameterLQRConstraint):
            raise TypeError(constraint)
        self._time = self.constraint.control_system.time
        super().__init__(self.objective, self.constraint,
                         lower_level_variables, higher_level_variables,
                         local_level_variables)

    @property
    def parameters(self):
        return self._parameters

    @property
    def state_input(self):
        return self._state_input

    @property
    def objective(self):
        return self._objective

    @property
    def constraint(self):
        return self._constraint

    def get_sensitivities(self, *args, **kwargs):
        boundary_value_problem = self.get_bvp()
        n_param = self._parameters.parameters.dimension
        return BVPSensitivities(boundary_value_problem, n_param, *args,
                                **kwargs)

    #TODO: Refactor parameter and non-parameter functions
    def get_bvp(self):
        nn = self._state_input.states.dimension
        nm = self._state_input.inputs.dimension
        dim = nm + 2 * nn
        n_param = self._parameters.parameters.dimension
        iweights = self.objective.integral_weights
        econtr = self.constraint.control_system.augmented_dae.e
        acontr = self.constraint.control_system.augmented_dae.a
        fcontr = self.constraint.control_system.augmented_dae.f

        zeros = jnp.zeros((nn, nn))
        zeros2 = jnp.zeros((nn + nm, nn + nm))
        zeros3 = jnp.zeros((nn + nm, ))
        zeros4 = jnp.zeros((nn + nm, 2 * nn + nm))
        zeros5 = jnp.zeros((nn, dim))
        zeros51 = jnp.zeros((nm, dim))
        zeros6 = jnp.zeros((nn, nm))

        @jax.jit
        def ecal(p, t):
            e = econtr(p, t)
            ecal = jnp.block([[zeros, e], [-e.T, zeros2]])
            return ecal

        @jax.jit
        def acal(p, t):
            a = acontr(p, t)
            iweight = iweights(p, t)
            acal = jnp.block([[zeros, a], [a.T, iweight]])
            return acal

        @jax.jit
        def fcal(p, t):
            fcal = jnp.block([fcontr(p, t), zeros3])
            return fcal

        @jax.jit
        def gamma_0(p):
            gamma_0_arr = jnp.block([[zeros, econtr(p, self._time.t_0)],
                                     [zeros4]])
            return gamma_0_arr

        @jax.jit
        def gamma_f(p):
            gamma_f_arr = jnp.block(
                [[zeros5],
                 [jnp.identity(nn),
                  self.objective.final_weight(p), zeros6], [zeros51]])
            return gamma_f_arr

        free_dae = self.constraint.control_system.free_dae

        @jax.jit
        def gamma_rhs(p):
            gamma_rhs_arr = jnp.block([
                free_dae.e(p, self._time.t_0)
                @ self.constraint.initial_value(p), zeros3
            ])
            return gamma_rhs_arr

        def z_gamma(p):
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
    def __init__(self, global_object, **kwargs):
        super().__init__(global_object, **kwargs)
        LQOptimalControl._init_local(self)


local_object_factory.register_localizer(ParameterDependentOptimalControl,
                                        AutomaticLQOptimalControl)
