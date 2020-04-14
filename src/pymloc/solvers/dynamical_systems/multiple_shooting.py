import logging
from copy import deepcopy

import numpy as np
import scipy.linalg as linalg
from pygelda.pygelda import Gelda

from ...misc import restack
from ...misc import unstack
from ...model.dynamical_system.boundary_value_problem import MultipleBoundaryValueProblem
from ...model.dynamical_system.flow_problem import LinearFlow
from ...model.dynamical_system.initial_value_problem import InitialValueProblem
from ...model.optimization.optimal_control import LQOptimalControl
from ...model.variables.time_function import Time
from ...solver_container import solver_container_factory
from ..base_solver import BaseSolver
from ..base_solver import TimeSolution

logger = logging.getLogger(__name__)


class MultipleShooting(BaseSolver):
    def __init__(self,
                 bvp: MultipleBoundaryValueProblem,
                 flow_problem: LinearFlow,
                 ivp_problem: InitialValueProblem,
                 shooting_nodes,
                 stepsize=1e-1,
                 *args,
                 **kwargs):
        if not isinstance(bvp, MultipleBoundaryValueProblem):
            raise TypeError(bvp)
        if not isinstance(flow_problem, LinearFlow):
            raise TypeError(flow_problem)
        if not isinstance(ivp_problem, InitialValueProblem):
            raise TypeError(ivp_problem)
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
        self._variables = self._dynamical_system.variables
        self._nn = self._dynamical_system.nn
        self._flow_problem = flow_problem
        self._ivp_problem = ivp_problem
        self._stepsize = stepsize
        logger.info('''MultipleShooting solver initialized with\n
        shooting_nodes: {}\n
        boundary_nodes: {}\n'''.format(shooting_nodes, self._bvp_nodes))
        super().__init__(*args, **kwargs)

    def _init_solver(self, time_interval, flow_abs_tol=None,
                     flow_rel_tol=None):
        self._set_t2s()
        self._set_d_as()
        self._init_flow_solver()
        self._flows = self._get_homogeneous_flows(flow_abs_tol, flow_rel_tol)
        self._set_gis()
        self._shooting_values = self._get_shooting_values()
        if time_interval is None:
            time_interval = self._variables.time
        self._time_interval = time_interval
        if time_interval.grid is None:
            time_interval.grid = self._shooting_nodes
        self._solution_time_grid = time_interval.grid

    def _init_flow_solver(self):
        time_interval = deepcopy(self._bvp.time_interval)
        time_interval.grid = self._shooting_nodes
        stepsize = 1. / (self._n_shooting_nodes - 1)
        logger.debug("Creating flow solver with time_interval: {}".format(
            time_interval.grid))
        self._flow_solver = solver_container_factory.get_solver_container(
            self._flow_problem).default_solver.solver(self._flow_problem,
                                                      time_interval,
                                                      stepsize,
                                                      rel_tol=self.rel_tol,
                                                      abs_tol=self.abs_tol)

    def _get_homogeneous_flows(self, abs_tol=None, rel_tol=None):
        flow_solver = self._flow_solver
        if abs_tol is not None:
            flow_solver.abs_tol = abs_tol
        if rel_tol is not None:
            flow_solver.rel_tol = rel_tol
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
        next_nodes = current_node_states - restack(lin_sys_sol,
                                                   current_node_states.shape)
        return next_nodes

    def _get_newton_rhs(self, current_node_states):
        current_x_d = self._get_x_d(current_node_states)
        boundary_node_states = current_x_d[..., self._boundary_indices]
        bound_res = self._bvp.boundary_values.residual(boundary_node_states)
        computed_node_states = self._forward_projected_nodes(
            current_node_states)
        diffs = current_node_states[..., 1:] - computed_node_states[..., :-1]
        res_b = unstack(diffs)
        return np.concatenate((res_b, bound_res), axis=0)

    def _forward_projected_nodes(self, node_values):
        solver = self._flow_solver
        lift_up = self._get_x_d(node_values)
        sol = solver.forward_solve_differential(lift_up)
        return self._project_values(sol)

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

    def _project_values(self, values):
        return np.einsum('ijr,i...r->j...r', self._t2s, values)

    def _run(self, time_interval=None, initial_guess=None, *args, **kwargs):
        self._init_solver(time_interval, *args, **kwargs)
        initial_guess = self._get_initial_guess(initial_guess)
        projected_values = self._project_values(initial_guess)
        for i in range(self.max_iter):
            residual = self._get_newton_rhs(projected_values)
            if self.abort(residual):
                break
            projected_values = self._newton_step(projected_values, residual)
        x_d = self._get_x_d(projected_values)
        full_node_values = x_d - np.einsum('ijr,j...r->i...r', self._das,
                                           x_d) - self._fas
        node_solution = TimeSolution(self._shooting_nodes, full_node_values)
        time_grid_solution = self._get_intermediate_values(node_solution)
        #TODO: Make plugin in time solvable
        self._bvp.variables.current_values = time_grid_solution
        return time_grid_solution, node_solution

    def _get_intermediate_values(self, node_solution):
        # this is rather slow, but it's an inherent disadvantage of the shooting approach
        logger.info(
            "Getting intermediate values on time grid of size: {}".format(
                self._solution_time_grid.size))
        idsize = self._solution_time_grid.size
        idx = np.append(
            np.searchsorted(self._solution_time_grid, node_solution.time_grid,
                            'left'), idsize)
        solution = np.zeros((*node_solution.solution.shape[:-1],
                             self._solution_time_grid.size))
        f_columns = self._bvp.boundary_values.n_inhom
        self._ivp_problem.init_solver(stepsize=self._stepsize,
                                      f_columns=f_columns,
                                      abs_tol=self.abs_tol,
                                      rel_tol=self.abs_tol)
        #TODO: Also compute backwards for better stability / difference to node points
        for i, node in enumerate(node_solution.time_grid):
            loweridx = idx[i]
            upperidx = idx[i + 1]
            grid = self._solution_time_grid[loweridx:upperidx]
            t0 = node
            if np.allclose(t0, grid[0], 1e-16):
                idx_increment = 0
            else:
                idx_increment = 1

            x0 = node_solution(node)
            tf = grid[-1]
            interval = Time(t0, tf, grid)
            x0_times = self._ivp_problem.solve(interval, x0)
            solution[..., loweridx:upperidx] = np.atleast_2d(
                x0_times.solution.T).T[..., idx_increment:]
        return TimeSolution(self._solution_time_grid, solution)

    def _get_x_d(self, projected_values):
        return np.einsum('ijr,j...r->i...r', self._t2s, projected_values)

    def _set_d_as(self):
        das = np.zeros((self._nn, self._nn, self._n_shooting_nodes), order='F')
        shape = self._dynamical_system.variables.shape
        fas = np.zeros((*shape, self._n_shooting_nodes), order='F')

        self._dynamical_system.init_rank()
        for i, node in enumerate(self._shooting_nodes):
            das[:, :, i] = self._dynamical_system.d_a(node)
            fas[..., i] = self._dynamical_system.f_a(node)
        self._das = das
        self._fas = fas


solver_container_factory.register_solver(MultipleBoundaryValueProblem,
                                         MultipleShooting,
                                         default=True)

solver_container_factory.register_solver(
    LQOptimalControl,
    MultipleShooting,
    default=True,
    creator_function=LQOptimalControl.get_bvp)
