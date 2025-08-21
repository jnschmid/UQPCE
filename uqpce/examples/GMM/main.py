import numpy as np
import openmdao.api as om
from uqpce.mdao.uqpcegroup import UQPCEGroup
from uqpce.mdao import interface
import os
import matplotlib.pyplot as plt

class SimpleQuadratic(om.ExplicitComponent):
    """
    Simple quadratic component with uncertain parameters
    f = a1*(x^2) + a2*x + a3
    where a1, a2, a3 are the uncertain parameters (GMM, uniform, lognormal)
    """
    def initialize(self):
        self.options.declare('vec_size', types=int)

    def setup(self):
        n = self.options['vec_size']
        arange = np.arange(n)

        # Design variable
        self.add_input('x', val=1.0)

        # Uncertain parameters
        self.add_input('a1', shape=(n,))  # GMM variable
        self.add_input('a2', shape=(n,))  # Uniform variable
        self.add_input('a3', shape=(n,))  # Lognormal variable

        # Output
        self.add_output('f', shape=(n,))

        # Declare partials
        self.declare_partials('f', 'x')
        self.declare_partials('f', ['a1', 'a2', 'a3'], rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        x = inputs['x']
        a1 = inputs['a1']
        a2 = inputs['a2']
        a3 = inputs['a3']

        outputs['f'] = a1 * (x**2) + a2 * x + a3

    def compute_partials(self, inputs, partials):
        n = self.options['vec_size']
        x = inputs['x']
        a1 = inputs['a1']
        a2 = inputs['a2']

        partials['f', 'x'] = (2 * a1 * x + a2).reshape(-1, 1)
        partials['f', 'a1'] = x**2
        partials['f', 'a2'] = x
        partials['f', 'a3'] = 1.0


class SimpleConstraint(om.ExplicitComponent):
    """
    Simple constraint: g = 5 - f
    """
    def initialize(self):
        self.options.declare('vec_size', types=int)

    def setup(self):
        n = self.options['vec_size']
        arange = np.arange(n)

        self.add_input('f', shape=(n,))
        self.add_output('g', shape=(n,))

        self.declare_partials('g', 'f', rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        outputs['g'] = 5.0 - inputs['f']

    def compute_partials(self, inputs, partials):
        partials['g', 'f'] = -1.0


if __name__ == '__main__':

    #---------------------------------------------------------------------------
    #                               Input Files
    #---------------------------------------------------------------------------

    script_dir = os.path.dirname(os.path.abspath(__file__))
    relative_yaml = 'input.yaml'
    relative_matrix = 'run_matrix.dat'
    input_file = os.path.join(script_dir, relative_yaml)
    matrix_file = os.path.join(script_dir, relative_matrix)

    #---------------------------------------------------------------------------
    #                   Setting up for UQPCE and design under uncertainty
    #---------------------------------------------------------------------------

    (
        var_basis, norm_sq, resampled_var_basis, 
        aleatory_cnt, epistemic_cnt, resp_cnt, order, variables, 
        sig, run_matrix
    ) = interface.initialize(input_file, matrix_file)
    
    prob = om.Problem()

    #---------------------------------------------------------------------------
    #                   Add Subsystems to Problem
    #---------------------------------------------------------------------------
    
    prob.model.add_subsystem(
        'quadratic',
        SimpleQuadratic(vec_size=resp_cnt),
        promotes_inputs=['x', 'a1', 'a2', 'a3'],
        promotes_outputs=['f']
    )

    prob.model.add_subsystem(
        'constraint',
        SimpleConstraint(vec_size=resp_cnt),
        promotes_inputs=['f'],
        promotes_outputs=['g']
    )

    #---------------------------------------------------------------------------
    #                   Add UQPCE Group to Problem
    #---------------------------------------------------------------------------
    prob.model.add_subsystem(
        'UQPCE',
        UQPCEGroup(
            significance=sig, var_basis=var_basis, norm_sq=norm_sq,
            resampled_var_basis=resampled_var_basis, tail='both',
            epistemic_cnt=epistemic_cnt, aleatory_cnt=aleatory_cnt,
            uncert_list=['f', 'g'], tanh_omega=1e-3,
            sample_ref0=[1, 1], sample_ref=[10, 10]
        ),
        promotes_inputs=['f', 'g'],
        promotes_outputs=[
            'f:mean', 'f:ci_lower', 'f:ci_upper', 'f:mean_plus_var', 'f:resampled_responses',
            'g:mean', 'g:ci_lower', 'g:ci_upper'
        ]
    )

    #---------------------------------------------------------------------------
    #                   Setting up the OpenMDAO Problem
    #---------------------------------------------------------------------------
    
    # Set up driver
    prob.driver = om.pyOptSparseDriver(optimizer='SNOPT')
    prob.driver.opt_settings['MAXIT'] = 50

    # Initial guess
    prob.model.set_input_defaults('x', 1.0)

    # Add design variable
    prob.model.add_design_var('x', lower=-3.0, upper=3.0)

    # Objective: minimize mean of f plus variance penalty
    prob.model.add_objective('f:mean_plus_var')

    # Constraint: g must be positive with 95% confidence
    prob.model.add_constraint('g:ci_lower', lower=0.0)

    prob.setup(force_alloc_complex=True)
    
    # Generate N2 diagram
    #om.n2(prob, show_browser=False, outfile='gmm_example_n2.html')

    # Use the UQPCE interface to set the uncertain parameters from the run matrix
    interface.set_vals(prob, variables, run_matrix)
    
    #---------------------------------------------------------------------------
    #                   Run the Problem and Print Results
    #---------------------------------------------------------------------------

    print("\n" + "="*60)
    print("GMM Example: Optimization Under Uncertainty")
    print("="*60)
    
    prob.run_model()

    print("\n" + "-"*60)
    print("OPTIMIZATION RESULTS:")
    print("-"*60)
    print(f"Optimal design variable x: {prob.get_val('x')[0]:.4f}")
    print(f"Objective (mean + variance): {prob.get_val('f:mean_plus_var')[0]:.4f}")
    print(f"Mean of f: {prob.get_val('f:mean')[0]:.4f}")
    print(f"CI Lower of f: {prob.get_val('f:ci_lower')[0]:.4f}")
    print(f"CI Upper of f: {prob.get_val('f:ci_upper')[0]:.4f}")
    print(f"Constraint g CI Lower: {prob.get_val('g:ci_lower')[0]:.4f}")
    print(f"Constraint g mean: {prob.get_val('g:mean')[0]:.4f}")

#---------------------------------------------------------------------------
    #                   Monte Carlo Verification
    #---------------------------------------------------------------------------
    
    print("\n" + "-"*60)
    print("MONTE CARLO VERIFICATION :")
    print("-"*60)
    
    # Get optimal x value
    x_opt = prob.get_val('x')[0]
    
    # Generate MC samples
    n_mc = 10_000_000
    
    # GMM samples (Variable 0) - Proper mixture sampling
    gmm_weights = np.array([0.3, 0.5, 0.2])
    gmm_means = np.array([-1.0, 0.0, 2.0])
    gmm_stdevs = np.array([0.3, 0.5, 0.4])
    
    # Select components based on weights
    component_indices = np.random.choice(3, size=n_mc, p=gmm_weights)
    gmm_samples = np.zeros(n_mc)
    
    for i in range(3):
        mask = component_indices == i
        n_samples = np.sum(mask)
        if n_samples > 0:
            gmm_samples[mask] = np.random.normal(
                gmm_means[i], gmm_stdevs[i], n_samples
            )
    
    # Clip to bounds (4 sigma from extremes)
    bounds_factor = 6
    gmm_low = float(np.min(gmm_means - bounds_factor * gmm_stdevs))
    gmm_high = float(np.max(gmm_means + bounds_factor * gmm_stdevs))
    gmm_samples = np.clip(gmm_samples, gmm_low, gmm_high)
    # Uniform samples (Variable 1)
    uniform_samples = np.random.uniform(-2.0, 2.0, n_mc)
    
    # Lognormal samples (Variable 2)
    # For lognormal: if X ~ LogNormal(mu, sigma), then ln(X) ~ Normal(mu, sigma)
    lognorm_samples = np.random.lognormal(mean=0, sigma=0.3, size=n_mc)
    
    # Evaluate function at optimal x
    f_mc = gmm_samples * (x_opt**2) + uniform_samples * x_opt + lognorm_samples
    g_mc = 5.0 - f_mc
    
    plt.hist(f_mc, bins=50, density=True, color='k',alpha=0.2)
    plt.hist(prob.get_val('f:resampled_responses'), bins=50, density=True, color='g',alpha=0.5)
    plt.savefig('f_hist_comparison')

    # Calculate statistics
    f_mean_mc = np.mean(f_mc)
    f_std_mc = np.std(f_mc)
    f_var_mc = np.var(f_mc)
    f_ci_lower_mc = np.percentile(f_mc, 2.5)
    f_ci_upper_mc = np.percentile(f_mc, 97.5)
    
    g_mean_mc = np.mean(g_mc)
    g_ci_lower_mc = np.percentile(g_mc, 2.5)
    g_ci_upper_mc = np.percentile(g_mc, 97.5)
    
    print(f"MC Mean of f: {f_mean_mc:.4f}")
    print(f"MC Std of f: {f_std_mc:.4f}")
    print(f"MC Variance of f: {f_var_mc:.4f}")
    print(f"MC 95% CI of f: [{f_ci_lower_mc:.4f}, {f_ci_upper_mc:.4f}]")
    print(f"MC Mean of g: {g_mean_mc:.4f}")
    print(f"MC 95% CI of g: [{g_ci_lower_mc:.4f}, {g_ci_upper_mc:.4f}]")
    print(f"MC Probability(g > 0): {np.mean(g_mc > 0):.3f}")
    
    # Compare with UQPCE results
    print("\n" + "-"*60)
    print("COMPARISON (UQPCE vs MC):")
    print("-"*60)
    
    uqpce_f_mean = prob.get_val('f:mean')[0]
    uqpce_f_ci_lower = prob.get_val('f:ci_lower')[0]
    uqpce_f_ci_upper = prob.get_val('f:ci_upper')[0]
    uqpce_g_mean = prob.get_val('g:mean')[0]
    uqpce_g_ci_lower = prob.get_val('g:ci_lower')[0]
    
    print(f"Mean of f - UQPCE: {uqpce_f_mean:.4f}, MC: {f_mean_mc:.4f}, "
          f"Diff: {abs(uqpce_f_mean - f_mean_mc):.4f}")
    print(f"CI Lower of f - UQPCE: {uqpce_f_ci_lower:.4f}, MC: {f_ci_lower_mc:.4f}, "
          f"Diff: {abs(uqpce_f_ci_lower - f_ci_lower_mc):.4f}")
    print(f"CI Upper of f - UQPCE: {uqpce_f_ci_upper:.4f}, MC: {f_ci_upper_mc:.4f}, "
          f"Diff: {abs(uqpce_f_ci_upper - f_ci_upper_mc):.4f}")
    print(f"Mean of g - UQPCE: {uqpce_g_mean:.4f}, MC: {g_mean_mc:.4f}, "
          f"Diff: {abs(uqpce_g_mean - g_mean_mc):.4f}")
    print(f"CI Lower of g - UQPCE: {uqpce_g_ci_lower:.4f}, MC: {g_ci_lower_mc:.4f}, "
          f"Diff: {abs(uqpce_g_ci_lower - g_ci_lower_mc):.4f}")
    
    # Verify theoretical properties of GMM
    print("\n" + "-"*60)
    print("GMM THEORETICAL VERIFICATION:")
    print("-"*60)
    
    theoretical_gmm_mean = np.sum(gmm_weights * gmm_means)
    empirical_gmm_mean = np.mean(gmm_samples)
    
    # Theoretical variance of GMM
    theoretical_gmm_var = 0
    for i in range(3):
        # Variance = E[X^2] - E[X]^2
        # For each component: w_i * (sigma_i^2 + mu_i^2)
        theoretical_gmm_var += gmm_weights[i] * (gmm_stdevs[i]**2 + gmm_means[i]**2)
    theoretical_gmm_var -= theoretical_gmm_mean**2
    
    empirical_gmm_var = np.var(gmm_samples)
    
    print(f"GMM Theoretical Mean: {theoretical_gmm_mean:.4f}, Empirical: {empirical_gmm_mean:.4f}")
    print(f"GMM Theoretical Variance: {theoretical_gmm_var:.4f}, Empirical: {empirical_gmm_var:.4f}")
    
    print("\n" + "="*60)