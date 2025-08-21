#!/usr/bin/env python
"""
Simple UQPCE Analysis with GMM Variable using OpenMDAO Integration
===================================================================

This example demonstrates:
1. Setting up uncertain variables including a GMM variable
2. Running UQPCE analysis through OpenMDAO without optimization
3. Comparing UQPCE statistics with Monte Carlo validation

The model function is: f = a1*x^2 + a2*x + a3
where:
  - a1 ~ GMM with 3 components (aleatory)
  - a2 ~ Uniform[-2, 2] (aleatory)
  - a3 ~ Lognormal(0.5, 0.3) (aleatory)
  - x is a deterministic design variable set to 2.0
"""

import numpy as np
import openmdao.api as om
from uqpce.mdao.uqpcegroup import UQPCEGroup
from uqpce.mdao import interface
import os


class SimpleQuadratic(om.ExplicitComponent):
    """
    Simple quadratic component with uncertain parameters
    f = a1*(x^2) + a2*x + a3
    where a1, a2, a3 are the uncertain parameters
    """
    def initialize(self):
        self.options.declare('vec_size', types=int)

    def setup(self):
        n = self.options['vec_size']
        arange = np.arange(n)

        # Design variable (deterministic)
        self.add_input('x', val=2.0)

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


def monte_carlo_validation(x_val=2.0, n_samples=100000, seed=42):
    """
    Run Monte Carlo simulation for validation
    """
    np.random.seed(seed)
    
    # GMM samples for a1
    gmm_weights = np.array([0.3, 0.5, 0.2])
    gmm_means = np.array([-1.0, 0.0, 2.0])
    gmm_stdevs = np.array([0.3, 0.5, 0.4])
    
    component_indices = np.random.choice(3, size=n_samples, p=gmm_weights)
    a1_samples = np.zeros(n_samples)
    
    for i in range(3):
        mask = component_indices == i
        n_comp = np.sum(mask)
        if n_comp > 0:
            a1_samples[mask] = np.random.normal(gmm_means[i], gmm_stdevs[i], n_comp)
    
    # Clip GMM to bounds
    bounds_factor = 4
    gmm_low = float(np.min(gmm_means - bounds_factor * gmm_stdevs))
    gmm_high = float(np.max(gmm_means + bounds_factor * gmm_stdevs))
    a1_samples = np.clip(a1_samples, gmm_low, gmm_high)
    
    # Uniform samples for a2
    a2_samples = np.random.uniform(-2.0, 2.0, n_samples)
    
    # Lognormal samples for a3
    a3_samples = np.random.lognormal(mean=0.5, sigma=0.3, size=n_samples)
    
    # Evaluate function
    f_samples = a1_samples * (x_val**2) + a2_samples * x_val + a3_samples
    
    return {
        'mean': np.mean(f_samples),
        'std': np.std(f_samples),
        'variance': np.var(f_samples),
        'percentile_2.5': np.percentile(f_samples, 2.5),
        'percentile_97.5': np.percentile(f_samples, 97.5),
        'samples': f_samples,
        'a1_mean': np.mean(a1_samples),
        'a1_var': np.var(a1_samples),
        'a2_mean': np.mean(a2_samples),
        'a2_var': np.var(a2_samples),
        'a3_mean': np.mean(a3_samples),
        'a3_var': np.var(a3_samples)
    }


if __name__ == '__main__':

    print("="*70)
    print("UQPCE Analysis with GMM Variable - OpenMDAO Integration")
    print("="*70)

    #---------------------------------------------------------------------------
    #                               Input Files
    #---------------------------------------------------------------------------

    # Create input.yaml
    input_yaml = """Variable 0:
  name: a1
  distribution: gaussian_mixture
  weights: [0.3, 0.5, 0.2]
  means: [-1.0, 0.0, 2.0]
  stdevs: [0.3, 0.5, 0.4]
  type: aleatory

Variable 1:
  name: a2
  distribution: uniform
  interval_low: -2.0
  interval_high: 2.0
  type: aleatory

Variable 2:
  name: a3
  distribution: lognormal
  mu: 0.5
  stdev: 0.3
  type: aleatory

Settings:
  order: 3
  backend: Agg
  aleat_samp_size: 10000
  significance: 0.05
"""

    # Write input.yaml
    with open('input.yaml', 'w') as f:
        f.write(input_yaml)

    # Create run_matrix.dat with proper sampling
    np.random.seed(42)
    n_samples = 40  # 2x oversampling for 20 terms (order 3, 3 variables)
    
    # Generate samples for each variable
    # GMM samples
    gmm_weights = [0.3, 0.5, 0.2]
    gmm_means = [-1.0, 0.0, 2.0]
    gmm_stdevs = [0.3, 0.5, 0.4]
    
    a1_samples = []
    for _ in range(n_samples):
        comp = np.random.choice(3, p=gmm_weights)
        sample = np.random.normal(gmm_means[comp], gmm_stdevs[comp])
        # Clip to bounds
        sample = np.clip(sample, -2.2, 2.8)
        a1_samples.append(sample)
    
    # Uniform samples
    a2_samples = np.random.uniform(-2.0, 2.0, n_samples)
    
    # Lognormal samples
    a3_samples = np.random.lognormal(0.5, 0.3, n_samples)
    
    # Combine into matrix
    run_matrix = np.column_stack([a1_samples, a2_samples, a3_samples])
    np.savetxt('run_matrix.dat', run_matrix)

    #---------------------------------------------------------------------------
    #                   Setting up for UQPCE
    #---------------------------------------------------------------------------

    print("\n1. Initializing UQPCE...")
    
    (
        var_basis, norm_sq, resampled_var_basis, 
        aleatory_cnt, epistemic_cnt, resp_cnt, order, variables, 
        sig, run_matrix
    ) = interface.initialize('input.yaml', 'run_matrix.dat')
    
    print(f"   Variables: {len(variables)}")
    print(f"   Aleatory: {aleatory_cnt}, Epistemic: {epistemic_cnt}")
    print(f"   PCE Order: {order}")
    print(f"   Response count (samples): {resp_cnt}")
    
    #---------------------------------------------------------------------------
    #                   Build OpenMDAO Problem
    #---------------------------------------------------------------------------
    
    print("\n2. Building OpenMDAO problem...")
    
    prob = om.Problem()
    
    # Add the quadratic component
    prob.model.add_subsystem(
        'quadratic',
        SimpleQuadratic(vec_size=resp_cnt),
        promotes_inputs=['x', 'a1', 'a2', 'a3'],
        promotes_outputs=['f']
    )
    
    # Add UQPCE analysis
    prob.model.add_subsystem(
        'UQPCE',
        UQPCEGroup(
            significance=sig, 
            var_basis=var_basis, 
            norm_sq=norm_sq,
            resampled_var_basis=resampled_var_basis, 
            tail='both',
            epistemic_cnt=epistemic_cnt, 
            aleatory_cnt=aleatory_cnt,
            uncert_list=['f'], 
            tanh_omega=1e-3,

        ),
        promotes_inputs=['f'],
        promotes_outputs=[
            'f:mean', 'f:variance', 'f:ci_lower', 'f:ci_upper',
            'f:resampled_responses'
        ]
    )
    
    # Setup the problem
    prob.setup()
    
    # Set design variable value
    prob.set_val('x', 2.0)
    
    # Set uncertain parameters from run matrix
    interface.set_vals(prob, variables, run_matrix)
    
    #---------------------------------------------------------------------------
    #                   Run Analysis (No Optimization)
    #---------------------------------------------------------------------------
    
    print("\n3. Running UQPCE analysis...")
    
    # Just run the model, no optimization
    prob.run_model()
    
    #---------------------------------------------------------------------------
    #                   Extract Results
    #---------------------------------------------------------------------------
    
    print("\n4. UQPCE Results:")
    print("-"*40)
    
    f_mean = prob.get_val('f:mean')[0]
    f_var = prob.get_val('f:variance')[0]
    f_std = np.sqrt(f_var)
    f_ci_lower = prob.get_val('f:ci_lower')[0]
    f_ci_upper = prob.get_val('f:ci_upper')[0]
    
    print(f"   Mean of f: {f_mean:.4f}")
    print(f"   Std dev of f: {f_std:.4f}")
    print(f"   Variance of f: {f_var:.4f}")
    print(f"   95% CI: [{f_ci_lower:.4f}, {f_ci_upper:.4f}]")
    
    # Get raw function evaluations for checking
    f_raw = prob.get_val('f')
    print(f"   Raw mean (from samples): {np.mean(f_raw):.4f}")
    print(f"   Raw std (from samples): {np.std(f_raw):.4f}")
    
    #---------------------------------------------------------------------------
    #                   Monte Carlo Validation
    #---------------------------------------------------------------------------
    
    print("\n5. Monte Carlo Validation (100,000 samples):")
    print("-"*40)
    
    mc_results = monte_carlo_validation(x_val=2.0, n_samples=100000)
    
    print(f"   Mean of f: {mc_results['mean']:.4f}")
    print(f"   Std dev of f: {mc_results['std']:.4f}")
    print(f"   Variance of f: {mc_results['variance']:.4f}")
    print(f"   95% CI: [{mc_results['percentile_2.5']:.4f}, "
          f"{mc_results['percentile_97.5']:.4f}]")
    
    #---------------------------------------------------------------------------
    #                   Comparison
    #---------------------------------------------------------------------------
    
    print("\n6. Comparison (UQPCE vs Monte Carlo):")
    print("-"*40)
    
    mean_error = abs(f_mean - mc_results['mean'])
    mean_error_pct = 100 * mean_error / abs(mc_results['mean'])
    
    var_error = abs(f_var - mc_results['variance'])
    var_error_pct = 100 * var_error / mc_results['variance']
    
    print(f"   Mean absolute error: {mean_error:.4f} ({mean_error_pct:.2f}%)")
    print(f"   Variance absolute error: {var_error:.4f} ({var_error_pct:.2f}%)")
    
    ci_width_uqpce = f_ci_upper - f_ci_lower
    ci_width_mc = mc_results['percentile_97.5'] - mc_results['percentile_2.5']
    print(f"   CI width - UQPCE: {ci_width_uqpce:.4f}")
    print(f"   CI width - MC: {ci_width_mc:.4f}")
    
    #---------------------------------------------------------------------------
    #                   Variable Statistics
    #---------------------------------------------------------------------------
    
    print("\n7. Variable Statistics (Monte Carlo):")
    print("-"*40)
    
    # Theoretical GMM statistics
    gmm_weights = np.array([0.3, 0.5, 0.2])
    gmm_means = np.array([-1.0, 0.0, 2.0])
    gmm_stdevs = np.array([0.3, 0.5, 0.4])
    
    theoretical_gmm_mean = np.sum(gmm_weights * gmm_means)
    theoretical_gmm_var = 0
    for i in range(3):
        theoretical_gmm_var += gmm_weights[i] * (gmm_stdevs[i]**2 + gmm_means[i]**2)
    theoretical_gmm_var -= theoretical_gmm_mean**2
    
    print(f"   a1 (GMM) theoretical mean: {theoretical_gmm_mean:.4f}")
    print(f"   a1 (GMM) empirical mean: {mc_results['a1_mean']:.4f}")
    print(f"   a1 (GMM) theoretical variance: {theoretical_gmm_var:.4f}")
    print(f"   a1 (GMM) empirical variance: {mc_results['a1_var']:.4f}")
    
    print(f"\n   a2 (Uniform) theoretical mean: 0.0000")
    print(f"   a2 (Uniform) empirical mean: {mc_results['a2_mean']:.4f}")
    print(f"   a2 (Uniform) theoretical variance: {4.0/3:.4f}")
    print(f"   a2 (Uniform) empirical variance: {mc_results['a2_var']:.4f}")
    
    # Lognormal theoretical values
    mu = 0.5
    sigma = 0.3
    lognorm_mean = np.exp(mu + sigma**2/2)
    lognorm_var = (np.exp(sigma**2) - 1) * np.exp(2*mu + sigma**2)
    
    print(f"\n   a3 (Lognormal) theoretical mean: {lognorm_mean:.4f}")
    print(f"   a3 (Lognormal) empirical mean: {mc_results['a3_mean']:.4f}")
    print(f"   a3 (Lognormal) theoretical variance: {lognorm_var:.4f}")
    print(f"   a3 (Lognormal) empirical variance: {mc_results['a3_var']:.4f}")
    
    #---------------------------------------------------------------------------
    #                   Summary
    #---------------------------------------------------------------------------
    
    print("\n" + "="*70)
    print("ANALYSIS SUMMARY")
    print("="*70)
    
    print(f"\nProblem: f = a1*x² + a2*x + a3, with x = 2.0")
    print(f"Variables:")
    print(f"  a1 ~ GMM(weights=[0.3,0.5,0.2], means=[-1,0,2], stds=[0.3,0.5,0.4])")
    print(f"  a2 ~ Uniform(-2, 2)")
    print(f"  a3 ~ Lognormal(μ=0.5, σ=0.3)")
    
    print(f"\nAccuracy (UQPCE vs 100k Monte Carlo):")
    print(f"  Mean error: {mean_error_pct:.2f}%")
    print(f"  Variance error: {var_error_pct:.2f}%")
    
    if mean_error_pct < 5 and var_error_pct < 10:
        print(f"\n✓ UQPCE analysis with GMM variable is working correctly!")
    else:
        print(f"\n⚠ Errors are larger than expected. Check sampling or PCE order.")
    
    print("="*70)
    '''
    # Clean up
    import os
    if os.path.exists('input.yaml'):
        os.remove('input.yaml')
    if os.path.exists('run_matrix.dat'):
        os.remove('run_matrix.dat')
    '''