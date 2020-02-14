import numpy as np
import scipy.linalg as linalg
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
        self._boundary_values = bvp.boundary_values
        self._bvp_nodes = bvp.time_points
        self._check_shooting_nodes()
        self._intervals = zip(self._shooting_nodes, self._shooting_nodes[1:])
        self._final_time = self._shooting_nodes[-1]
        self._inner_nodes = shooting_nodes[1:-1]
        self._inhomogeinity = bvp.inhomogeinity
        self._dynamical_system = bvp.dynamical_system
        self._ivp_solver = ivp_solver
        self._nn = self._dynamical_system.nn
        super().__init__(stepsize)

    def _get_shooting_values(self, t2s):
        n = self._nn
        shooting_values = np.zeros((self._n_shooting_nodes, n, n))
        j = 0
        for i, node in enumerate(self._shooting_nodes):
            if node in self._bvp_nodes:
                shooting_values[i, ...] = self._boundary_values[j]
                j += 1
        projected_values = np.einsum('rij,rjk->rik', shooting_values, t2s)
        return projected_values.reshape(n * self._n_shooting_nodes, n).T

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
        gis = self._compute_gis(t2s, flows)
        jis = self._compute_jis()
        bc = self._boundary_values
        dim = self._dynamical_system.rank * self._n_shooting_nodes
        shooting_matrix = np.zeros((dim, dim))
        #TODO: Inefficient, probably needs low level implementation
        diag = linalg.block_diag(*(gis[i, ...] for i in range(gis.shape[0])))
        shooting_matrix[:diag.shape[0], :diag.shape[1]] = diag
        for i in range(self._n_shooting_nodes)[:-1]:
            size = self._dynamical_system.rank * i
            sizep1 = self._dynamical_system.rank * (i + 1)
            sizep2 = self._dynamical_system.rank * (i + 2)
            shooting_matrix[size:sizep1, sizep1:sizep2] = jis[i, ...]

        shooting_values = self._get_shooting_values(t2s)
        shooting_matrix = np.block([[shooting_matrix], [shooting_values]])
        return shooting_matrix

    def _get_mesh(self, stepsize):
        mesh = np.concatenate((np.arange(t_lower, t_upper, stepsize)
                               for t_lower, t_upper in self._intervals))
        return mesh

    def _compute_flows(self):
        n = self._nn
        flows = np.zeros((self._n_shooting_nodes - 1, n, n))
        #TODO: Paralellize
        for i, (t_i, t_ip1) in enumerate(self._intervals):
            flows[i, :, :] = self._compute_flow(t_i, t_ip1)
        return flows

    def _compute_flow(self, t_i, t_ip1):
        n = self._nn
        flow = np.zeros((n, n), order='F')
        #TODO: Potentially Slow! implement a flow routine at low level routine or parallelize
        for i, unit_vector in enumerate(np.identity(n)):
            flow[:, i] = self._ivp_solver.run(t_i, t_ip1, unit_vector)[:, -1]
        return flow

    def _compute_gis(self, t2, flows):
        t2_1 = t2[1:, :, :]
        t2_e = t2[:-1, :, :]
        gis = np.einsum('rji,rjk,rkl->ril', t2_1, flows, t2_e)
        return gis

    def _compute_jis(self):
        #TODO: only works in the linear case
        rank = self._dynamical_system.rank
        return np.array(self._n_shooting_nodes * (np.identity(rank), ))

    def _compute_t2(self):
        t2s = np.array([
            self._dynamical_system.get_t2(node)
            for node in self._shooting_nodes
        ])
        return t2s
