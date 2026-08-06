import unittest

import numpy as np
import openmdao.api as om
import openmdao.utils.assert_utils as om_assert

from uqpce.mdao.differencecomp import DifferenceComp


class TestDifferenceComp(unittest.TestCase):
    def setUp(self):

        prob = om.Problem(reports=False)

        prob.model.add_subsystem(
            'comp', DifferenceComp(vec_size=5),
            promotes_inputs=['*'], promotes_outputs=['*']
        )

        prob.setup(force_alloc_complex=True)
        prob.set_val(
            'actual_responses', np.array([
                12.16415154, 9.01613385, 7.20322875, 8.36029185, 8.540021
            ])
        )
        prob.set_val(
            'pred_responses', np.array([
                12.26415154, 9.11613385, 7.30322875, 8.46029185, 8.640021
            ])
        )

        prob.run_model()
        self.partials = prob.check_partials(out_stream=None, method='cs')
        self.prob = prob

    def test_partials(self):
        partials = self.partials
        om_assert.assert_check_partials(partials, atol=1e-6, rtol=1e-6)

    def test_difference(self):
        act_err = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
        calc_err = self.prob.get_val('comp.difference')
        self.assertTrue(
            np.isclose(act_err, calc_err).all(),
            msg='DifferenceComp is not calculating difference correctly'
        )


if __name__ == '__main__':

    np.random.seed(33)

    suite = unittest.TestSuite()
    unittest.main()
