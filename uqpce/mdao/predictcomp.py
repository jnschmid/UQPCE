import numpy as np
import openmdao.api as om


class PredictComp(om.ExplicitComponent):
    """
    Class definition for the PredictComp.

    A PredictComp predicts the response values from the PCE model.
    """
    def initialize(self):
        self.options.declare(
            'var_basis', types=np.ndarray,
            desc='The variable basis of the PCE model.'
        )
        self._no_check_partials = True

    def setup(self):
        """
        Setup the PredictComp.
        """
        var_basis = self.options['var_basis']
        resp_cnt, term_cnt = var_basis.shape

        self.add_input('matrix_coeffs', shape=(term_cnt,))
        self.add_output('pred_responses', shape=(resp_cnt,))

        self.declare_partials(
            of='pred_responses', wrt='matrix_coeffs',
            val=var_basis
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        """
        Compute the PredictComp.
        """
        var_basis = self.options['var_basis']
        outputs['pred_responses'] = np.matmul(
            var_basis, inputs['matrix_coeffs']
        )
