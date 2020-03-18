import numpy as np

from ..dynamical_system.boundary_value_problem import BoundaryValueProblem
from ..dynamical_system.boundary_value_problem import BoundaryValues
from ..dynamical_system.flow_problem import LinearFlow
from ..dynamical_system.initial_value_problem import InitialValueProblem
from ..dynamical_system.representations import LinearFlowRepresentation
from ..variables.container import StateVariablesContainer
from .constraints.lqr import LQRConstraint
from .local_optimization import LocalOptimizationObject
from .objectives.lqr import LQRObjective


class LQOptimalControl(LocalOptimizationObject):
    def __init__(self, objective: LQRObjective, constraint: LQRConstraint,
                 variables):
        super().__init__(objective, constraint, variables)
        self._init_local()

    def _init_local(self):
        self._nm = self.constraint.control_system.nm
        self._nn = self.constraint.control_system.nn
        self._time = self.constraint.control_system.time

    def get_bvp(self):
        dim = self._nm + 2 * self._nn
        acal_arr = np.zeros((dim, dim))
        ecal_arr = np.zeros((dim, dim))
        fcal_arr = np.zeros((dim, ))
        iweights = self.objective.integral_weights
        econtr = self.constraint.control_system.augmented_dae.e
        acontr = self.constraint.control_system.augmented_dae.a
        fcontr = self.constraint.control_system.augmented_dae.f

        def ecal(t):
            e = econtr(t)
            ecal_arr[:self._nn, self._nn:] = e
            ecal_arr[self._nn:, :self._nn] = -e.T
            return ecal_arr

        def acal(t):
            a = acontr(t)
            acal_arr[:self._nn, self._nn:] = a
            acal_arr[self._nn:, :self._nn] = a.T
            acal_arr[self._nn:, self._nn:] = iweights(t)
            return acal_arr

        def fcal(t):
            fcal_arr[:self._nn] = fcontr(t)
            return fcal_arr

        gamma_0 = np.zeros((dim, dim))
        gamma_f = np.zeros((dim, dim))
        gamma_0[:self._nn, self._nn:] = econtr(self._time.t_0)
        gamma_f[self._nn:2 * self._nn, :self._nn] = np.identity(self._nn)
        gamma_f[self._nn:2 * self._nn, self._nn:2 *
                self._nn] = self.objective.final_weight
        gamma_rhs = np.zeros((dim, ))
        gamma_rhs[:self._nn] = self.constraint.initial_value

        variables = StateVariablesContainer(dim)
        dyn_sys = LinearFlowRepresentation(variables, ecal, acal, fcal, dim)
        bvs = BoundaryValues(gamma_0, gamma_f, gamma_rhs)
        flow_prob = LinearFlow(self._time, dyn_sys)
        ivp_prob = InitialValueProblem(self.constraint.initial_value,
                                       self._time, dyn_sys)
        shooting_nodes = np.linspace(self._time.t_0, self._time.t_f, 3)
        return BoundaryValueProblem(self._time, dyn_sys,
                                    bvs), flow_prob, ivp_prob, shooting_nodes
