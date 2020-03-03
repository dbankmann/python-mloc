import numpy as np
import scipy.linalg as linalg
from pygelda.pygelda import Gelda

from ...model.dynamical_system.boundary_value_problem import MultipleBoundaryValueProblem
from ...solver_container import solver_container_factory
from ..base_solver import BaseSolver


class MultipleShooting(BaseSolver):
    def __init__(self, bvp, flow_problem, shooting_nodes, stepsize):
        if not isinstance(bvp, MultipleBoundaryValueProblem):
            raise TypeError(bvp)
        self._bvp = bvp
        self._shooting_nodes = shooting_nodes
        self._n_shooting_nodes = len(shooting_nodes)
        self._boundary_values = bvp.boundary_values.boundary_values
        self._bvp_nodes = bvp.nodes
        self._check_shooting_nodes()
        self._intervals = zip(self._shooting_nodes, self._shooting_nodes[1:])
        self._final_time = self._shooting_nodes[-1]
        self._inner_nodes = shooting_nodes[1:-1]
        self._dynamical_system = bvp.dynamical_system
        self._nn = self._dynamical_system.nn
        self._flow_problem = flow_problem
        super().__init__(stepsize)

    def _init_solver(self):
        self._set_t2s()
        self._set_d_as()
        self._flows = self._get_homogeneous_flows()
        self._set_gis()
        self._shooting_values = self._get_shooting_values()

    def _get_homogeneous_flows(self):
        time_interval = self._bvp.time_interval
        time_interval.grid = self._shooting_nodes
        stepsize = 1. / (self._n_shooting_nodes - 1)
        flow_solver = solver_container_factory.get_solver_container(
            self._flow_problem).default_solver(self._flow_problem,
                                               time_interval, stepsize)
        flow_solver.abs_tol = 1e-3
        flow_solver.rel_tol = 1e-6
        return flow_solver.get_homogeneous_flows()

    def _get_shooting_values(self):
        n = self._nn
        rank = self._dynamical_system.rank
        shooting_values = np.zeros((n, n, self._n_shooting_nodes), order='F')
        j = 0
        indices = np.array([], dtype=int)
        for i, node in enumerate(self._shooting_nodes):
            if node in self._bvp_nodes:
                shooting_values[..., i] = self._boundary_values[..., j]
                indices = np.append(indices, [i])
                j += 1
        z_gamma = self._bvp.boundary_values.z_gamma
        projected_values = np.einsum('ai, ajr,jkr->ikr', z_gamma,
                                     shooting_values, self._t2s)
        self._boundary_indices = indices
        return projected_values.reshape(rank,
                                        rank * self._n_shooting_nodes,
                                        order='F')

    def _check_shooting_nodes(self):
        for node in self._bvp_nodes:
            if not node in self._shooting_nodes:
                raise ValueError(
                    "Shooting nodes {}\nhave to include boundary value nodes {}"
                    .format(self._shooting_nodes, self._bvp_nodes))

    @property
    def bvp(self):
        return self._bvp

    def _get_shooting_matrix(self):
        gis = self._gis
        jis = self._compute_jis()
        bc = self._boundary_values
        dim = self._dynamical_system.rank * self._n_shooting_nodes
        shooting_matrix = np.zeros((dim, dim), order='F')
        #TODO: Inefficient, probably needs low level implementation
        diag = linalg.block_diag(*(gis[..., i] for i in range(gis.shape[-1])))
        shooting_matrix[:diag.shape[0], :diag.shape[1]] = -diag
        for i in range(self._n_shooting_nodes)[:-1]:
            size = self._dynamical_system.rank * i
            sizep1 = self._dynamical_system.rank * (i + 1)
            sizep2 = self._dynamical_system.rank * (i + 2)
            shooting_matrix[size:sizep1, sizep1:sizep2] = jis[..., i]
        shooting_matrix[sizep1:, :] = self._shooting_values
        return shooting_matrix

    def _get_mesh(self, stepsize):
        mesh = np.concatenate((np.arange(t_lower, t_upper, stepsize)
                               for t_lower, t_upper in self._intervals))
        return mesh

    def _set_gis(self):
        t2 = self._t2s
        t2_1 = t2[:, :, 1:]
        t2_e = t2[:, :, :-1]
        gis = np.einsum('jir,jkr,klr->ilr', t2_1, self._flows, t2_e)
        self._gis = gis

    def _compute_jis(self):
        #TODO: only works in the linear case
        rank = self._dynamical_system.rank
        ntemp = self._n_shooting_nodes - 1
        return np.array(ntemp * (np.identity(rank), )).T

    def _set_t2s(self):
        rank = self._dynamical_system.rank
        t2s = np.zeros((self._nn, rank, self._n_shooting_nodes), order='F')
        for i, node in enumerate(self._shooting_nodes):
            t2s[:, :, i] = self._dynamical_system.t2(node)
        self._t2s = t2s

    def _newton_step(self, current_node_states, rhs):
        shooting_matrix = self._get_shooting_matrix()
        lin_sys_sol = np.linalg.solve(shooting_matrix, rhs)
        next_nodes = current_node_states - self._restack(
            lin_sys_sol, current_node_states.shape)
        return next_nodes

    @staticmethod
    def _unstack(array):
        return np.moveaxis(array, -1, 0).reshape(-1, *array.shape[1:-1])

    @staticmethod
    def _restack(array, shape):
        *shapefirst, shapem2, shapem1 = shape
        restacked = np.einsum('ij...->j...i',
                              array.reshape(shapem1, *shapefirst, shapem2))
        return restacked

    def _get_newton_rhs(self, current_node_states):
        current_x_d = self._get_x_d(current_node_states)
        boundary_node_states = current_x_d[..., self._boundary_indices]
        bound_res = self._bvp.boundary_values.residual(boundary_node_states)
        diffs = current_node_states[..., 1:] - np.einsum(
            'ijr,j...r->i...r', self._gis, current_node_states[..., :-1])
        res_b = self._unstack(diffs)
        return np.concatenate((res_b, bound_res), axis=0)

    def _get_initial_guess(self, initial_guess=None):
        shape = self._dynamical_system.variables.shape + (
            self._n_shooting_nodes, )
        if initial_guess is None:
            initial_guess = np.zeros(shape, order='F')
        elif initial_guess.shape != shape:
            raise ValueError(
                "initial_guess.shape is {} but should equal {}".format(
                    initial_guess.shape, shape))
        return initial_guess

    def run(self, initial_guess=None):
        self._init_solver()
        initial_guess = self._get_initial_guess(initial_guess)
        projected_values = np.einsum('ijr,i...r->j...r', self._t2s,
                                     initial_guess)

        for i in range(self.max_iter):
            residual = self._get_newton_rhs(projected_values)
            if self.abort(residual):
                break
            projected_values = self._newton_step(projected_values, residual)
        x_d = self._get_x_d(projected_values)
        full_node_values = x_d + np.einsum('ijr,j...r->i...r', self._das, x_d)
        return self._shooting_nodes, full_node_values

    def _get_x_d(self, projected_values):
        return np.einsum('ijr,j...r->i...r', self._t2s, projected_values)

    def _set_d_as(self):
        das = np.zeros((self._nn, self._nn, self._n_shooting_nodes), order='F')
        for i, node in enumerate(self._shooting_nodes):
            das[:, :, i] = self._dynamical_system.d_a(node)
        self._das = das


solver_container_factory.register_solver(MultipleBoundaryValueProblem,
                                         MultipleShooting,
                                         default=True)
