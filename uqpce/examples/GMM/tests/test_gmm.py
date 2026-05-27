import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials

from uqpce.examples.GMM.gmm import (
    QuadraticFunction,
    monte_carlo_uq,
    run_deterministic_with_uqpce,
)


class TestQuadraticFunction(unittest.TestCase):
    def setUp(self):
        np.random.seed(33)

        prob = om.Problem()
        prob.model.add_subsystem(
            'comp', QuadraticFunction(vec_size=4),
            promotes_inputs=['*'], promotes_outputs=['*']
        )
        prob.setup(force_alloc_complex=True)
        prob.set_val('x', 0)
        prob.set_val('y', 0)
        prob.set_val('a1', np.array([-2.1, -0.25, 0.2, 3.5]))
        prob.set_val('a2', np.array([-1.5, -0.5, 0.5, 1.5]))
        prob.set_val('a3', np.array([0.83, 0.15, -0.18, -0.95]))
        prob.run_model()
        self.partials = prob.check_partials(out_stream=None, method='cs')

        self.calc_ans = prob.get_val('f')[0]

    def test_compute(self):
        truth = 10.27389

        self.assertTrue(
            np.isclose(self.calc_ans, truth),
            msg='QuadraticFunction `compute` is not correct.'
        )

    def test_partials(self):
        assert_check_partials(self.partials, atol=1e-6, rtol=1e-6)


class TestMonteCarlo(unittest.TestCase):
    def setUp(self):
        pass

    def test_monte_carlo_uq(self):
        f = monte_carlo_uq(x=0, y=0, n_samples=1_000)

        self.assertTrue(
            f.min() > 7.0,
            msg='Function `monte_carlo_uq` is not correct.'
        )
        self.assertTrue(
            f.max() < 22.0,
            msg='Function `monte_carlo_uq` is not correct.'
        )

    def test_run_deterministic_with_uqpce(self):
        input_file = 'uqpce/examples/GMM/input.yaml'
        matrix_file = 'uqpce/examples/GMM/run_matrix.dat'
        x_det, y_det, f_det = run_deterministic_with_uqpce(input_file, matrix_file)

        self.assertTrue(
            np.isclose(x_det, 2),
            msg='Function `run_deterministic_with_uqpce` is not correct.'
        )
        self.assertTrue(
            np.isclose(y_det, 2),
            msg='Function `run_deterministic_with_uqpce` is not correct.'
        )
        self.assertTrue(
            np.isclose(f_det, 0),
            msg='Function `run_deterministic_with_uqpce` is not correct.'
        )

if __name__ == '__main__':
    suite = unittest.TestSuite()
    unittest.main()
