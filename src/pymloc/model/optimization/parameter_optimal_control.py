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

        zeros = np.zeros((nn, nn))
        zeros2 = np.zeros((nn + nm, nn + nm))

        def ecal(p, t):
            e = econtr(p, t)
            ecal = jnp.block([[zeros, e], [-e.T, zeros2]])
            return ecal

        def acal(p, t):
            a = acontr(p, t)
            iweight = iweights(p, t)
            acal = jnp.block([[zeros, a], [a.T, iweight]])
            return acal

        def fcal(p, t):
            zeros3 = np.zeros((nn + nm, ))
            fcal = np.block([fcontr(p, t), zeros3])
            return fcal

        gamma_0_arr = np.zeros((dim, dim))
        gamma_f_arr = np.zeros((dim, dim))
        gamma_rhs_arr = np.zeros((dim, ))

        def gamma_0(p):
            gamma_0_arr[:nn, nn:] = econtr(p, self._time.t_0)
            return gamma_0_arr

        def gamma_f(p):
            gamma_f_arr[nn:2 * nn, :nn] = np.identity(nn)
            gamma_f_arr[nn:2 * nn, nn:2 * nn] = self.objective.final_weight(p)
            return gamma_f_arr

        def gamma_rhs(p):
            gamma_rhs_arr[:nn] = self.constraint.initial_value(p)
            return gamma_rhs_arr

        variables = StateVariablesContainer(dim)
        bvs = ParameterBoundaryValues(self.lower_level_variables,
                                      self.higher_level_variables, variables,
                                      gamma_0, gamma_f, gamma_rhs, dim,
                                      n_param)
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
