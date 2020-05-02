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
import numpy as np

from pymloc.model.dynamical_system.parameter_dae import jac_jax_reshaped


class TestParamaterDAE:
    def test_compute_theta_e(self, e_lin_param_dae):
        jac = jac_jax_reshaped(e_lin_param_dae, (3, 3))
        np.testing.assert_equal(jac(3., 3.), np.zeros((3, 3, 1)))

    def test_compute_theta_a(self, a_lin_param_dae):
        jac = jac_jax_reshaped(a_lin_param_dae, (3, 3))
        np.testing.assert_equal(jac(3., 3.)[..., 0], np.diag((0, -1., 0)))

    def test_compute_theta_f(self, f_lin_param_dae):
        jac = jac_jax_reshaped(f_lin_param_dae, (3, ))
        np.testing.assert_equal(jac(3., 3.), np.zeros((3, 1)))
