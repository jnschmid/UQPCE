import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials

from uqpce.examples.paraboloid.paraboloid import Paraboloid


class TestParaboloid(unittest.TestCase):
    def setUp(self):
        np.random.seed(33)

        prob = om.Problem()
        prob.model.add_subsystem(
            'comp', Paraboloid(vec_size=4),
            promotes_inputs=['*'], promotes_outputs=['*']
        )
        prob.setup(force_alloc_complex=True)
        prob.set_val('uncerta', np.array([1.25, 1.5, 1.75, 2.0]))
        prob.set_val('uncertb', np.array([5.0, 4.0, 3.0, 2.0]))
        prob.set_val('desx', 8)
        prob.set_val('desy', 12)
        prob.run_model()
        self.partials = prob.check_partials(out_stream=None, method='cs')

        self.calc_ans = prob.get_val('f_abxy')[0]

    def test_compute(self):
        truth = 4742

        self.assertTrue(
            np.isclose(self.calc_ans, truth),
            msg='Paraboloid `compute` is not correct.'
        )

    def test_partials(self):
        assert_check_partials(self.partials, atol=1e-6, rtol=1e-6)


if __name__ == '__main__':
    suite = unittest.TestSuite()
    unittest.main()
