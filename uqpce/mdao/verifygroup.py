import numpy as np
import openmdao.api as om

from uqpce.mdao.differencecomp import DifferenceComp
from uqpce.mdao.predictcomp import PredictComp


class ErrorGroup(om.Group):
    """
    Class definition for the ErrorGroup.

    A ErrorGroup outputs the error between the true responses and the
    responses predicted from the PCE model.
    """
    def initialize(self):
        self.options.declare(
            'var_basis', types=(np.ndarray, type(None)), allow_none=True,
            default=None, desc='The variable basis of the PCE model.'
        )

    def setup(self):
        """
        Setup the ErrorGroup.
        """
        var_basis = self.options['var_basis']
        vec_size = var_basis.shape[0]

        self.add_subsystem(
            'pred_resp_comp', PredictComp(var_basis=var_basis),
            promotes_inputs=['matrix_coeffs'],
            promotes_outputs=['pred_responses']
        )

        self.add_subsystem(
            'error_comp', DifferenceComp(vec_size=vec_size),
            promotes_inputs=['pred_responses', 'actual_responses'],
            promotes_outputs=[('difference', 'error')]
        )


class VerifyGroup(ErrorGroup):
    """
    Class definition for the VerifyGroup.

    A VerifyGroup wraps around the ErrorGroup to provide clarity to
    users. This group performs the same function as the ErrorGroup, but it
    predicts/compares for responses that were not used to build the model.
    """


if __name__ == '__main__':
    var_basis = np.array([
        [1, -1.69821781e+00, 9.67008130e-01],
        [1, 2.64654707e-01, -5.54940304e-01],
        [1, 9.04884946e-02, -9.27743660e-01],
        [1, -1.27260262e+00, -3.13090952e-02],
        [1, -7.19942644e-01, 4.53624248e-01,]
    ])
    prob = om.Problem(reports=False)

    prob.model.add_subsystem(
        'comp', ErrorGroup(var_basis=var_basis),
        promotes_inputs=['*'], promotes_outputs=['*']
    )

    prob.setup(force_alloc_complex=True)
    prob.set_val(
        'matrix_coeffs', np.array([[12.26415154], [9.11613385], [7.30322875]])
    )
    prob.set_val(
        'actual_responses', np.array([
            12.16415154, 9.01613385, 7.20322875, 8.36029185, 8.540021
        ])
    )

    prob.run_model()
    prob.check_partials(method='cs', compact_print=True)
