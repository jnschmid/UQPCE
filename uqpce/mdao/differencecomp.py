import numpy as np
import openmdao.api as om


class DifferenceComp(om.ExplicitComponent):
    """
    Class definition for the DifferenceComp.

    A DifferenceComp calculates the element-wise difference between two vectors.
    """
    def initialize(self):
        self.options.declare(
            'vec_size', types=(int, float),
            desc='The size of the input response vectors.'
        )
        self._no_check_partials = True

    def setup(self):
        """
        Setup the DifferenceComp.
        """
        vec_size = self.options['vec_size']
        arange = np.arange(vec_size)

        self.add_input('pred_responses', shape=(vec_size,))
        self.add_input('actual_responses', shape=(vec_size,))
        self.add_output('difference', shape=(vec_size,))

        self.declare_partials(
            of='difference', wrt='pred_responses', val=1, cols=arange, rows=arange
        )
        self.declare_partials(
            of='difference', wrt='actual_responses', val=-1, cols=arange, rows=arange
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        """
        Compute the DifferenceComp.
        """
        outputs['difference'] = inputs['pred_responses'] - inputs['actual_responses']
