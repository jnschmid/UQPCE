import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials

from uqpce.examples.dymos_projectile.widthCI import WidthCI


class TestCost(unittest.TestCase):
    def setUp(self):
        prob = om.Problem(reports=None)
        prob.model.add_subsystem(
            'dist', WidthCI(), promotes_inputs=['*'], promotes_outputs=['*']
        )

        prob.setup()
        prob.set_val('x_out:ci_lower', 132.58208526)
        prob.set_val('x_out:ci_upper', 167.91431536)

        prob.run_model()
        self.partials = prob.check_partials(out_stream=None)
        self.prob = prob

    def test_partials(self):
        assert_check_partials(self.partials, atol=1e-6, rtol=1e-6)

    def test_compute(self):
        ub = self.prob.get_val('x_out:ci_upper')
        lb = self.prob.get_val('x_out:ci_lower')
        calc = ub - lb
        diff_truth = 167.91431536 - 132.58208526

        self.assertTrue(
            np.isclose(calc, diff_truth), msg="CI width error."
        )


if __name__ == '__main__':
    suite = unittest.TestSuite()
    unittest.main()