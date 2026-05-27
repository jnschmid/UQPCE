import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials

from uqpce.examples.dymos_projectile.cost import Cost


class TestCost(unittest.TestCase):
    def setUp(self):
        # Number of sample points
        resp_cnt = 12
        prob = om.Problem(reports=None)
        prob.model.add_subsystem(
            'cost', Cost(num_samples=resp_cnt), promotes_inputs=['*'], promotes_outputs=['*']
        )

        prob.setup()
        prob.set_val('v', 80)
        prob.set_val('m', 12)

        prob.run_model()
        self.partials = prob.check_partials(out_stream=None)
        self.prob = prob

    def test_partials(self):
        assert_check_partials(self.partials, atol=1e-6, rtol=1e-6)

    def test_compute(self):
        ke = self.prob.get_val('cost')

        # Truth value computed with KE = m*v**2 == (12)*(80)**2
        ke_truth = 38400

        self.assertTrue(
            np.isclose(ke, ke_truth), msg="Area computation error."
        )


if __name__ == '__main__':
    suite = unittest.TestSuite()
    unittest.main()