import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials

from uqpce.mdao.sobolcomp import SobolComp
from uqpce.pce._helpers import calc_sobols, create_total_sobols


class TestSobolComp(unittest.TestCase):
    def setUp(self):
        np.random.seed(33)

    def test_sobol_comp_standalone(self):
        """Test SobolComp as standalone component."""

        # Create simple test data
        n_vars = 2

        # Mock norm_sq values (positive values for valid variance)
        norm_sq = np.array([[1.0], [1.0], [1.0], [1.0]])  # Shape (n_terms, 1)

        # Model matrix: which variables appear in each term
        # Term 0: intercept (no variables)
        # Term 1: variable 0
        # Term 2: variable 1
        # Term 3: variables 0 and 1 (interaction)
        model_matrix = np.array([
            [0, 0],  # intercept
            [1, 0],  # var 0 only
            [0, 1],  # var 1 only
            [1, 1],  # interaction
        ])

        # Test coefficients
        matrix_coeffs = np.array([1.0, 0.5, 0.3, 0.2])

        # Create problem
        prob = om.Problem()
        prob.model = om.Group()

        # Add component
        prob.model.add_subsystem(
            'sobol_comp',
            SobolComp(norm_sq=norm_sq, model_matrix=model_matrix),
            promotes=['*']
        )

        prob.setup(force_alloc_complex=True)
        prob.set_val('matrix_coeffs', matrix_coeffs)
        prob.run_model()

        # Get results
        sobols = prob.get_val('sobols')
        total_sobols = prob.get_val('total_sobols')

        # Verify using direct calculation
        expected_sobols = calc_sobols(matrix_coeffs, norm_sq)
        expected_sobols_2d = expected_sobols.reshape(-1, 1)
        expected_total = create_total_sobols(n_vars, model_matrix, expected_sobols_2d)

        self.assertTrue(
            np.allclose(sobols, expected_sobols),
            msg="Individual Sobols mismatch!"
        )
        self.assertTrue(
            np.allclose(total_sobols, expected_total.flatten()),
            msg="Total Sobols mismatch!"
        )

    def test_sobol_comp_derivatives(self):
        """Test SobolComp analytic derivatives against complex step."""

        # Same setup as above
        norm_sq = np.array([[1.0], [1.0], [1.0], [1.0]])
        model_matrix = np.array([
            [0, 0],
            [1, 0],
            [0, 1],
            [1, 1],
        ])
        matrix_coeffs = np.array([1.0, 0.5, 0.3, 0.2])

        # Create problem
        prob = om.Problem()
        prob.model = om.Group()

        prob.model.add_subsystem(
            'sobol_comp',
            SobolComp(norm_sq=norm_sq, model_matrix=model_matrix),
            promotes=['*']
        )

        prob.setup(force_alloc_complex=True)
        prob.set_val('matrix_coeffs', matrix_coeffs)
        prob.run_model()

        # Check analytic partials against complex step
        data = prob.check_partials(out_stream=None, method='cs')

        # Use reasonable tolerances for complex step
        assert_check_partials(data, atol=1e-6, rtol=1e-6)


if __name__ == '__main__':

    np.random.seed(33)

    suite = unittest.TestSuite()
    unittest.main()
