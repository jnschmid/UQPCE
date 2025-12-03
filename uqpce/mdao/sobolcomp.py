"""OpenMDAO component for computing Sobol sensitivity indices."""
import numpy as np
import openmdao.api as om
from uqpce.pce._helpers import calc_sobols, create_total_sobols


class SobolComp(om.ExplicitComponent):
    """
    Computes Sobol sensitivity indices using existing UQPCE functions.

    This component wraps uqpce.pce._helpers.calc_sobols() and
    create_total_sobols() to expose Sobol indices as OpenMDAO outputs.

    Individual Sobol indices show the variance contribution of each PCE term.
    Total Sobol indices show the variance contribution of each uncertain variable
    (including interaction effects).

    Mathematical formulation:
        variance = Σ(coeff[i]² × norm_sq[i]) for i > 0
        sobol[i-1] = (coeff[i]² × norm_sq[i]) / variance
        total_sobol[j] = Σ sobol[i-1] where variable j appears in term i
    """

    def initialize(self):
        """Declare options for the component."""
        self.options.declare(
            'norm_sq', types=np.ndarray, allow_none=False,
            desc='Norm squared for PCE terms'
        )
        self.options.declare(
            'model_matrix', types=np.ndarray, allow_none=False,
            desc='Interaction matrix for computing total Sobols. '
                 'Shape: (n_terms, n_vars). Entry [i,j] indicates if variable j '
                 'appears in PCE term i.'
        )

    def setup(self):
        """Set up inputs and outputs for the component."""
        norm_sq = self.options['norm_sq']
        model_matrix = self.options['model_matrix']

        n_terms = len(norm_sq)
        n_vars = model_matrix.shape[1]
        n_sobols = n_terms - 1  # Exclude intercept

        # Input: PCE coefficients from CoefficientsComp
        self.add_input(
            'matrix_coeffs', shape=(n_terms,),
            desc='PCE coefficients from CoefficientsComp'
        )

        # Output: Individual Sobol indices (one per PCE term, excluding intercept)
        self.add_output(
            'sobols', shape=(n_sobols,),
            desc='Individual Sobol indices (variance contribution per PCE term)'
        )

        # Output: Total Sobol indices (one per uncertain variable)
        self.add_output(
            'total_sobols', shape=(n_vars,),
            desc='Total Sobol indices (variance contribution per variable, including interactions)'
        )

        # Declare partials with analytic derivatives
        # Individual Sobols depend on all coefficients (through variance in denominator)
        self.declare_partials('sobols', 'matrix_coeffs', method='exact')

        # Total Sobols are linear combinations of individual Sobols
        # The Jacobian is sparse based on model_matrix structure
        # For simplicity, declare as dense (could optimize to sparse later)
        self.declare_partials('total_sobols', 'matrix_coeffs', method='exact')

    def compute(self, inputs, outputs):
        """
        Compute Sobols using existing UQPCE functions.

        Uses:
        - uqpce.pce._helpers.calc_sobols() for individual Sobols
        - uqpce.pce._helpers.create_total_sobols() for total Sobols

        Supports complex step by taking real part of complex inputs.
        """
        matrix_coeffs = inputs['matrix_coeffs']
        norm_sq = self.options['norm_sq']
        model_matrix = self.options['model_matrix']

        # Handle complex step: UQPCE functions don't support complex,
        # but Sobols are real-valued functions of real coefficients
        if np.iscomplexobj(matrix_coeffs):
            matrix_coeffs = matrix_coeffs.real

        # Compute individual Sobols using existing tested UQPCE function
        sobols = calc_sobols(matrix_coeffs, norm_sq)
        outputs['sobols'] = sobols

        # Compute total Sobols using existing tested UQPCE function
        var_count = model_matrix.shape[1]

        # create_total_sobols expects sobols to be 2D: (n_terms, n_responses)
        # Reshape 1D sobols to 2D (n_terms, 1) for single response
        sobols_2d = sobols.reshape(-1, 1)
        total_sobols = create_total_sobols(var_count, model_matrix, sobols_2d)

        # Flatten to 1D array for output
        outputs['total_sobols'] = total_sobols.flatten()

    def compute_partials(self, inputs, partials):
        """
        Compute analytic derivatives of Sobols with respect to coefficients.

        For sobol[i-1] = (coeff[i]² × norm_sq[i]) / variance:

        ∂sobol[i-1]/∂coeff[i] = 2×coeff[i]×norm_sq[i]/variance
                                 - sobol[i-1]×2×coeff[i]×norm_sq[i]/variance
                               = 2×coeff[i]×norm_sq[i]/variance × (1 - sobol[i-1])

        ∂sobol[i-1]/∂coeff[j] (j≠i) = -sobol[i-1]×2×coeff[j]×norm_sq[j]/variance
                                      = -2×coeff[j]×norm_sq[j]×sobol[i-1]/variance

        For total_sobol[var] = Σ sobol[i] where model_matrix[i, var] != 0:

        ∂total_sobol[var]/∂coeff[j] = Σ ∂sobol[i]/∂coeff[j] for relevant i
        """
        matrix_coeffs = inputs['matrix_coeffs']
        norm_sq = self.options['norm_sq']
        model_matrix = self.options['model_matrix']

        n_terms = len(matrix_coeffs)
        n_sobols = n_terms - 1

        # Handle complex step: take real part
        if np.iscomplexobj(matrix_coeffs):
            matrix_coeffs = matrix_coeffs.real

        # Compute variance (denominator of Sobol formula)
        # variance = Σ(coeff[i]² × norm_sq[i]) for i > 0
        # norm_sq is 2D (n_terms, 1), extract scalar values
        variance = 0.0
        for i in range(1, n_terms):
            variance += matrix_coeffs[i]**2 * float(norm_sq[i])

        # Compute individual Sobols for derivative calculation
        sobols = np.zeros(n_sobols)
        for i in range(1, n_terms):
            sobols[i-1] = (matrix_coeffs[i]**2 * float(norm_sq[i])) / variance

        # Initialize Jacobian matrix for d(sobols)/d(matrix_coeffs)
        jac_sobols = np.zeros((n_sobols, n_terms))

        # Derivatives of individual Sobols
        for i in range(1, n_terms):  # For each Sobol index
            sobol_idx = i - 1

            for j in range(n_terms):  # For each coefficient
                if j == 0:
                    # Intercept doesn't affect Sobols
                    jac_sobols[sobol_idx, j] = 0.0
                elif j == i:
                    # Derivative with respect to own coefficient
                    # ∂sobol[i]/∂coeff[i] = 2×coeff[i]×norm_sq[i]/variance × (1 - sobol[i])
                    jac_sobols[sobol_idx, j] = (
                        2.0 * matrix_coeffs[i] * float(norm_sq[i]) / variance * (1.0 - sobols[sobol_idx])
                    )
                else:
                    # Derivative with respect to other coefficients (through variance)
                    # ∂sobol[i]/∂coeff[j] = -2×coeff[j]×norm_sq[j]×sobol[i]/variance
                    jac_sobols[sobol_idx, j] = (
                        -2.0 * matrix_coeffs[j] * float(norm_sq[j]) * sobols[sobol_idx] / variance
                    )

        partials['sobols', 'matrix_coeffs'] = jac_sobols

        # Derivatives of total Sobols
        # total_sobol[var] = Σ sobol[i-1] where model_matrix[i, var] != 0
        # ∂total_sobol[var]/∂coeff[j] = Σ ∂sobol[i-1]/∂coeff[j] for relevant i

        n_vars = model_matrix.shape[1]
        jac_total = np.zeros((n_vars, n_terms))

        for var_idx in range(n_vars):  # For each variable
            for term_idx in range(1, n_terms):  # For each PCE term (excluding intercept)
                if model_matrix[term_idx, var_idx] != 0:
                    # This term contributes to this variable's total Sobol
                    # Add the derivatives from individual Sobol
                    jac_total[var_idx, :] += jac_sobols[term_idx-1, :]

        partials['total_sobols', 'matrix_coeffs'] = jac_total
