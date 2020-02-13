import numpy as np
from pygelda.pygelda import Gelda

from ...model.dynamical_system.boundary_value_problem import MultipleBoundaryValueProblem
from ..base_solver import BaseSolver


class MultipleShooting(BaseSolver):
    def __init__(self, bvp, ivp_solver, shooting_nodes, stepsize):
        if not isinstance(bvp, MultipleBoundaryValueProblem):
            raise TypeError(bvp)
        self._bvp = bvp
        self._shooting_nodes = shooting_nodes
        self._n_shooting_nodes = len(shooting_nodes)
        self._bvp_nodes = bvp.boundary_values
        self._check_shooting_nodes()
        self._bvp_timepoints = bvp.time_points
        self._intervals = zip(self._shooting_nodes, self._shooting_nodes[1:])
        self._final_time = self._shooting_nodes[-1]
        self._inner_nodes = shooting_nodes[1:-1]
        self._inhomogeinity = bvp.inhomogeinity
        self._dynamical_system = bvp.dynamical_system
        self._ivp_solver = ivp_solver
        self._nn = self._dynamical_system.nn
        super().__init__(stepsize)

    def _check_shooting_nodes(self):
        for node in self._bvp_nodes:
            if not node in self._shooting_nodes:
                raise ValueError(
                    "Shooting nodes {}\nhave to include boundary value nodes {}"
                    .format(self._shooting_nodes, self._bvp_nodes))

    @property
    def bvp(self):
        return self._bvp

    def run(self, initial_values, jump_conditions):
        self._build_shooting_matrix()

    def _build_shooting_matrix(self):
        t2s = self._compute_t2()
        flows = self._compute_flows()
        gis = self._compute_gis()
        jis = self._compute_ji()
        bc = self._compute_boundary_coefficients()
        idnd = np.identity(self.nnd)
        upper_diag = np.block(
            [np.zeros(), linalg.block_diag([idnd] * self.nnodes)])
        pass

    def _get_mesh(self, stepsize):
        mesh = np.concatenate((np.arange(t_lower, t_upper, stepsize)
                               for t_lower, t_upper in self._intervals))
        return mesh

    def _compute_flows(self):
        n = self._nn
        flows = np.zeros((n, n, self._n_shooting_nodes - 1))
        #TODO: Paralellize
        for i, (t_i, t_ip1) in enumerate(self._intervals):
            flows[:, :, i] = self._compute_flow(t_i, t_ip1)
        return flows

    def _compute_flow(self, t_i, t_ip1):
        n = self._nn
        flow = np.zeros((n, n), order='F')
        #TODO: Potentially Slow! implement a flow routine at low level routine or parallelize
        for i, unit_vector in enumerate(np.identity(n)):
            flow[:, i] = self._ivp_solver.run(t_i, t_ip1, unit_vector)[:, -1]
        return flow

    def _compute_gi(self):
        for i, node in enumerate(self.nodes):
            pass

    def _compute_t2(self):
        t2s = [
            self._dynamical_system.get_t2(node)
            for node in self._shooting_nodes
        ]
        return t2s

    def _compute_inner_t2_tilde(self, t2s):
        t2_tilde_inner = t2s[1:-1]  #only works in linear case
        return t2_tilde_inner
