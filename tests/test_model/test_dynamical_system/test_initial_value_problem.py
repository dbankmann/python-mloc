import pytest


class TestInitialValueProblem:
    def test_solve(self, initial_value_problem):
        initial_value_problem.init_solver(stepsize=1e-3)
        initial_value_problem.solve()
