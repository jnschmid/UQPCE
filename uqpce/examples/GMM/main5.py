#!/usr/bin/env python
"""
GMM-PCE Verification Script
===========================
Verifies PCE with GMM variables against Monte Carlo using analytical functions.
Based on main2.py and main3.py examples.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import ks_2samp, wasserstein_distance
import openmdao.api as om
from uqpce.mdao.uqpcegroup import UQPCEGroup
from uqpce.mdao import interface
import os


class SimpleQuadratic(om.ExplicitComponent):
    """
    Simple quadratic component: f = a1*(x^2) + a2*x + a3
    """
    def initialize(self):
        self.options.declare('vec_size', types=int)

    def setup(self):
        n = self.options['vec_size']
        arange = np.arange(n)

        self.add_input('x', val=2.0)
        self.add_input('a1', shape=(n,))  # GMM variable
        self.add_input('a2', shape=(n,))  # Uniform variable
        self.add_input('a3', shape=(n,))  # Lognormal variable
        self.add_output('f', shape=(n,))

        self.declare_partials('f', 'x')
        self.declare_partials('f', ['a1', 'a2', 'a3'], rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        x = inputs['x']
        outputs['f'] = inputs['a1'] * (x**2) + inputs['a2'] * x + inputs['a3']

    def compute_partials(self, inputs, partials):
        x = inputs['x']
        partials['f', 'x'] = (2 * inputs['a1'] * x + inputs['a2']).reshape(-1, 1)
        partials['f', 'a1'] = x**2
        partials['f', 'a2'] = x
        partials['f', 'a3'] = 1.0


def monte_carlo_validation(variables, x_val=2.0, n_samples=100000):
    """Run Monte Carlo simulation for validation"""
    np.random.seed(42)
    
    # Generate samples for each variable
    mc_samples = np.zeros((n_samples, len(variables)))
    
    for i, var in enumerate(variables):
        if hasattr(var, 'weights'):  # GMM variable
            # Proper GMM sampling
            components = np.random.choice(len(var.weights), size=n_samples, p=var.weights)
            samples = np.zeros(n_samples)
            for j, (mu, sigma) in enumerate(zip(var.means, var.stdevs)):
                mask = components == j
                n_comp = np.sum(mask)
                if n_comp > 0:
                    samples[mask] = np.random.normal(mu, sigma, n_comp)
            mc_samples[:, i] = samples
        else:
            # Use variable's generate_samples method
            mc_samples[:, i] = var.generate_samples(n_samples)
    
    # Evaluate function
    f_samples = mc_samples[:, 0] * (x_val**2) + mc_samples[:, 1] * x_val + mc_samples[:, 2]
    
    return {
        'mean': np.mean(f_samples),
        'std': np.std(f_samples),
        'variance': np.var(f_samples),
        'percentile_2.5': np.percentile(f_samples, 2.5),
        'percentile_97.5': np.percentile(f_samples, 97.5),
        'samples': f_samples,
        'mc_samples': mc_samples
    }


def verify_gmm_sampling(variables, n_test=10000):
    """Verify GMM sampling distribution"""
    print("\n" + "="*60)
    print("GMM SAMPLING VERIFICATION")
    print("="*60)
    
    results = {}
    
    for i, var in enumerate(variables):
        if hasattr(var, 'weights'):  # GMM variable
            print(f"\nVariable: {var.name}")
            print(f"  Weights: {var.weights}")
            print(f"  Means: {var.means}")
            print(f"  Stdevs: {var.stdevs}")
            
            # Generate samples
            samples = var.generate_samples(n_test)
            
            # Generate reference GMM samples
            components = np.random.choice(len(var.weights), size=n_test, p=var.weights)
            ref_samples = np.zeros(n_test)
            for j, (mu, sigma) in enumerate(zip(var.means, var.stdevs)):
                mask = components == j
                n_comp = np.sum(mask)
                if n_comp > 0:
                    ref_samples[mask] = np.random.normal(mu, sigma, n_comp)
            
            # Statistical tests
            ks_stat, ks_pval = ks_2samp(samples, ref_samples)
            w_distance = wasserstein_distance(samples, ref_samples)
            
            # Moment comparison
            theo_mean = np.sum(var.weights * var.means)
            theo_var = np.sum(var.weights * (var.stdevs**2 + var.means**2)) - theo_mean**2
            emp_mean = np.mean(samples)
            emp_var = np.var(samples)
            
            print(f"  KS test p-value: {ks_pval:.4f} (>0.05 is good)")
            print(f"  Wasserstein distance: {w_distance:.6f}")
            print(f"  Mean - Theory: {theo_mean:.4f}, Empirical: {emp_mean:.4f}")
            print(f"  Var  - Theory: {theo_var:.4f}, Empirical: {emp_var:.4f}")
            
            results[var.name] = {
                'ks_pvalue': ks_pval,
                'wasserstein': w_distance,
                'mean_error': abs(emp_mean - theo_mean),
                'var_error': abs(emp_var - theo_var)
            }
    
    return results


def test_orthogonality(variables):
    """Test orthogonality of polynomials"""
    print("\n" + "="*60)
    print("ORTHOGONALITY VERIFICATION")
    print("="*60)
    
    for var in variables:
        if hasattr(var, 'weights'):  # GMM variable
            print(f"\nVariable: {var.name}")
            
            if hasattr(var, 'var_orthopoly_vect'):
                basis = var.var_orthopoly_vect
                n_poly = min(3, len(basis))
                
                # Numerical integration
                x = np.linspace(var.interval_low, var.interval_high, 1000)
                dx = x[1] - x[0]
                
                # GMM PDF
                pdf = np.zeros_like(x)
                for w, mu, sig in zip(var.weights, var.means, var.stdevs):
                    pdf += w * stats.norm.pdf(x, mu, sig)
                
                print("  Orthogonality (<Pi,Pj> should be ~0 for i≠j):")
                for i in range(n_poly):
                    for j in range(i+1, n_poly):
                        try:
                            from sympy import lambdify
                            pi_func = lambdify(var.x, basis[i], 'numpy')
                            pj_func = lambdify(var.x, basis[j], 'numpy')
                            pi_vals = pi_func(x)
                            pj_vals = pj_func(x)
                            inner = np.sum(pi_vals * pj_vals * pdf) * dx
                            print(f"    <P{i},P{j}>: {inner:.6f}")
                        except:
                            print(f"    <P{i},P{j}>: Could not evaluate")


def run_verification():
    """Main verification routine"""
    
    print("="*70)
    print("UQPCE GMM VERIFICATION - ANALYTICAL FUNCTION TEST")
    print("="*70)
    
    # Initialize UQPCE
    input_file = 'input.yaml'
    matrix_file = 'run_matrix.dat'
    
    if not os.path.exists(input_file):
        print("Error: input.yaml not found")
        return False
    
    # Initialize using interface
    (
        var_basis, norm_sq, resampled_var_basis,
        aleatory_cnt, epistemic_cnt, resp_cnt, order, variables,
        sig, run_matrix
    ) = interface.initialize(input_file, matrix_file)
    
    print(f"\nVariables: {len(variables)}")
    print(f"Aleatory: {aleatory_cnt}, Epistemic: {epistemic_cnt}")
    print(f"PCE Order: {order}")
    print(f"Samples: {resp_cnt}")
    
    # Test 1: GMM Sampling
    sampling_results = verify_gmm_sampling(variables)
    
    # Test 2: Orthogonality
    test_orthogonality(variables)
    
    # Build OpenMDAO problem
    prob = om.Problem()
    
    prob.model.add_subsystem(
        'quadratic',
        SimpleQuadratic(vec_size=resp_cnt),
        promotes_inputs=['x', 'a1', 'a2', 'a3'],
        promotes_outputs=['f']
    )
    
    prob.model.add_subsystem(
        'UQPCE',
        UQPCEGroup(
            significance=sig, var_basis=var_basis, norm_sq=norm_sq,
            resampled_var_basis=resampled_var_basis, tail='both',
            epistemic_cnt=epistemic_cnt, aleatory_cnt=aleatory_cnt,
            uncert_list=['f'], tanh_omega=1e-3
        ),
        promotes_inputs=['f'],
        promotes_outputs=['f:mean', 'f:variance', 'f:ci_lower', 'f:ci_upper']
    )
    
    prob.setup()
    prob.set_val('x', 2.0)
    interface.set_vals(prob, variables, run_matrix)
    
    # Run UQPCE analysis
    print("\n" + "="*60)
    print("UQPCE ANALYSIS")
    print("="*60)
    
    prob.run_model()
    
    uqpce_mean = prob.get_val('f:mean')[0]
    uqpce_var = prob.get_val('f:variance')[0]
    uqpce_ci_lower = prob.get_val('f:ci_lower')[0]
    uqpce_ci_upper = prob.get_val('f:ci_upper')[0]
    
    print(f"Mean: {uqpce_mean:.4f}")
    print(f"Variance: {uqpce_var:.4f}")
    print(f"95% CI: [{uqpce_ci_lower:.4f}, {uqpce_ci_upper:.4f}]")
    
    # Monte Carlo validation
    print("\n" + "="*60)
    print("MONTE CARLO VALIDATION")
    print("="*60)
    
    mc_results = monte_carlo_validation(variables, x_val=2.0, n_samples=100000)
    
    print(f"Mean: {mc_results['mean']:.4f}")
    print(f"Std: {mc_results['std']:.4f}")
    print(f"95% CI: [{mc_results['percentile_2.5']:.4f}, {mc_results['percentile_97.5']:.4f}]")
    
    # Comparison
    print("\n" + "="*60)
    print("COMPARISON (UQPCE vs MC)")
    print("="*60)
    
    mean_error = abs(uqpce_mean - mc_results['mean']) / abs(mc_results['mean']) * 100
    var_error = abs(uqpce_var - mc_results['variance']) / abs(mc_results['variance']) * 100
    
    print(f"Mean relative error: {mean_error:.2f}%")
    print(f"Variance relative error: {var_error:.2f}%")
    
    # Generate plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Output distribution
    axes[0, 0].hist(mc_results['samples'], bins=100, density=True, alpha=0.7, label='Monte Carlo')
    axes[0, 0].axvline(uqpce_mean, color='red', linestyle='--', label='UQPCE Mean')
    axes[0, 0].axvline(mc_results['mean'], color='blue', linestyle='--', label='MC Mean')
    axes[0, 0].set_xlabel('Response Value')
    axes[0, 0].set_ylabel('Probability Density')
    axes[0, 0].set_title('Output Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: GMM variable distribution
    for i, var in enumerate(variables):
        if hasattr(var, 'weights'):
            samples = var.generate_samples(10000)
            axes[0, 1].hist(samples, bins=50, density=True, alpha=0.7, label=f'{var.name} samples')
            
            x_plot = np.linspace(samples.min(), samples.max(), 500)
            pdf = np.zeros_like(x_plot)
            for w, mu, sig in zip(var.weights, var.means, var.stdevs):
                pdf += w * stats.norm.pdf(x_plot, mu, sig)
            axes[0, 1].plot(x_plot, pdf, 'r-', lw=2, label=f'{var.name} PDF')
            
    axes[0, 1].set_xlabel('Value')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('GMM Variable Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Q-Q plot for each variable
    for i, var in enumerate(variables):
        if i < 3:  # Only first 3 variables
            empirical = run_matrix[:, i]
            theoretical = var.generate_samples(len(empirical))
            
            emp_quantiles = np.percentile(empirical, np.linspace(0, 100, 50))
            theo_quantiles = np.percentile(theoretical, np.linspace(0, 100, 50))
            
            axes[1, i//2].scatter(theo_quantiles, emp_quantiles, alpha=0.5, label=var.name)
    
    for i in range(2):
        min_val = axes[1, i].get_xlim()[0]
        max_val = axes[1, i].get_xlim()[1]
        axes[1, i].plot([min_val, max_val], [min_val, max_val], 'r--', lw=1)
        axes[1, i].set_xlabel('Theoretical Quantiles')
        axes[1, i].set_ylabel('Empirical Quantiles')
        axes[1, i].set_title('Q-Q Plot')
        axes[1, i].legend()
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gmm_pce_verification.png', dpi=150)
    plt.show()
    
    # Summary
    print("\n" + "="*60)
    if mean_error < 5 and var_error < 10:
        print("✓ VERIFICATION PASSED")
    else:
        print("✗ VERIFICATION FAILED")
    print("="*60)
    
    return mean_error < 5 and var_error < 10


if __name__ == '__main__':
    success = run_verification()
    exit(0 if success else 1)