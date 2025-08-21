#!/usr/bin/env python
"""
GMM-PCE Verification Script
===========================
Verifies PCE with GMM variables against Monte Carlo using analytical functions.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
from uqpce.pce.pce import PCE
from uqpce.pce.io import read_input_file


def analytical_function(x):
    """Test function: f = a1*x_val^2 + a2*x_val + a3 where x_val=2"""
    x_val = 2.0
    return x[0] * (x_val**2) + x[1] * x_val + x[2]


def monte_carlo_simulation(pce_model, n_samples=100000):
    """Run Monte Carlo simulation using PCE variable sampling"""
    np.random.seed(42)
    
    # Generate samples using PCE's sampling
    X_mc = pce_model.sample(count=n_samples)
    
    # Evaluate analytical function
    y_mc = np.array([analytical_function(x) for x in X_mc])
    
    return X_mc, y_mc


def run_verification():
    """Main verification routine"""
    
    print("="*70)
    print("GMM-PCE VERIFICATION")
    print("="*70)
    
    # Read input file
    if not os.path.exists('input.yaml'):
        print("Error: input.yaml not found")
        return False
    
    var_dict, settings = read_input_file('input.yaml')
    
    # Create PCE model
    print("\n1. Setting up PCE model...")
    pce = PCE(**settings)
    for key, value in var_dict.items():
        pce.add_variable(**value)
    
    print(f"   Variables: {len(pce.variables)}")
    print(f"   Order: {pce.order}")
    
    # Generate training samples
    n_terms = int(np.math.factorial(len(pce.variables) + pce.order) / 
                 (np.math.factorial(len(pce.variables)) * np.math.factorial(pce.order)))
    n_train = 2 * n_terms  # 2x oversampling
    
    print(f"\n2. Generating {n_train} training samples...")
    X_train = pce.sample(count=n_train)
    y_train = np.array([analytical_function(x) for x in X_train])
    
    # Fit PCE
    print("\n3. Fitting PCE model...")
    coeffs = pce.fit(X_train, y_train)
    print(f"   Coefficients: {len(coeffs)}")
    
    # Extract mean and variance (handling array vs scalar)
    if hasattr(pce, 'mean'):
        pce_mean = float(pce.mean) if np.isscalar(pce.mean) else float(pce.mean[0])
        print(f"   Mean: {pce_mean:.4f}")
    
    if hasattr(pce, 'variance'):
        pce_var = float(pce.variance) if np.isscalar(pce.variance) else float(pce.variance[0])
        print(f"   Variance: {pce_var:.4f}")
    else:
        pce_mean = np.mean(y_train)
        pce_var = np.var(y_train)
        print(f"   Mean (empirical): {pce_mean:.4f}")
        print(f"   Variance (empirical): {pce_var:.4f}")
    
    # Monte Carlo validation
    print("\n4. Running Monte Carlo validation...")
    X_mc, y_mc = monte_carlo_simulation(pce, n_samples=100000)
    
    mc_mean = np.mean(y_mc)
    mc_var = np.var(y_mc)
    mc_std = np.std(y_mc)
    mc_percentiles = np.percentile(y_mc, [2.5, 25, 50, 75, 97.5])
    
    # PCE predictions on MC samples
    print("\n5. Evaluating PCE on MC samples...")
    y_pce = pce.predict(X_mc).flatten()
    
    pce_mean_empirical = np.mean(y_pce)
    pce_var_empirical = np.var(y_pce)
    pce_std_empirical = np.std(y_pce)
    
    # Comparison
    print("\n" + "-"*60)
    print("STATISTICS COMPARISON")
    print("-"*60)
    print(f"{'Metric':<20} {'Monte Carlo':<15} {'PCE Direct':<15} {'PCE Empirical':<15}")
    print("-"*60)
    print(f"{'Mean':<20} {mc_mean:<15.4f} {pce_mean:<15.4f} {pce_mean_empirical:<15.4f}")
    print(f"{'Variance':<20} {mc_var:<15.4f} {pce_var:<15.4f} {pce_var_empirical:<15.4f}")
    print(f"{'Std Dev':<20} {mc_std:<15.4f} {np.sqrt(pce_var):<15.4f} {pce_std_empirical:<15.4f}")
    
    # Test GMM variable properties
    print("\n" + "-"*60)
    print("GMM VARIABLE VERIFICATION")
    print("-"*60)
    
    for i, var in enumerate(pce.variables):
        if hasattr(var, 'weights'):  # GMM variable
            print(f"\nVariable: {var.name}")
            weights = np.array(var.weights) if hasattr(var.weights, '__len__') else [var.weights]
            means = np.array(var.means) if hasattr(var.means, '__len__') else [var.means]
            stdevs = np.array(var.stdevs) if hasattr(var.stdevs, '__len__') else [var.stdevs]
            
            print(f"  Weights: {weights}")
            print(f"  Means: {means}")
            print(f"  Stdevs: {stdevs}")
            
            # Theoretical statistics
            theo_mean = np.sum(weights * means)
            theo_var = np.sum(weights * (stdevs**2 + means**2)) - theo_mean**2
            
            # Empirical from training samples
            emp_mean = np.mean(X_train[:, i])
            emp_var = np.var(X_train[:, i])
            
            print(f"  Theoretical - Mean: {theo_mean:.4f}, Var: {theo_var:.4f}")
            print(f"  Empirical   - Mean: {emp_mean:.4f}, Var: {emp_var:.4f}")
            
            # Test orthogonality
            if hasattr(var, 'var_orthopoly_vect'):
                print(f"  Orthogonal polynomials: {len(var.var_orthopoly_vect)} terms")
    
    # Generate plots
    print("\n6. Generating plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Output distribution comparison
    axes[0, 0].hist(y_mc, bins=100, density=True, alpha=0.7, label='Monte Carlo', color='blue')
    axes[0, 0].hist(y_pce, bins=100, density=True, alpha=0.7, label='PCE', color='red')
    axes[0, 0].axvline(mc_mean, color='blue', linestyle='--', label=f'MC Mean={mc_mean:.2f}')
    axes[0, 0].axvline(pce_mean, color='red', linestyle='--', label=f'PCE Mean={pce_mean:.2f}')
    axes[0, 0].set_xlabel('Response Value')
    axes[0, 0].set_ylabel('Probability Density')
    axes[0, 0].set_title('Output Distribution Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Q-Q plot
    mc_quantiles = np.percentile(y_mc, np.linspace(1, 99, 99))
    pce_quantiles = np.percentile(y_pce, np.linspace(1, 99, 99))
    
    axes[0, 1].scatter(mc_quantiles, pce_quantiles, alpha=0.5, s=5)
    min_val = min(mc_quantiles.min(), pce_quantiles.min())
    max_val = max(mc_quantiles.max(), pce_quantiles.max())
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    axes[0, 1].set_xlabel('Monte Carlo Quantiles')
    axes[0, 1].set_ylabel('PCE Quantiles')
    axes[0, 1].set_title('Q-Q Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Variable distributions
    for i, var in enumerate(pce.variables):
        if i < 3:  # First 3 variables
            samples = X_train[:, i]
            axes[1, i//2].hist(samples, bins=30, density=True, alpha=0.7, label=var.name)
            
            if hasattr(var, 'weights'):  # GMM
                x_plot = np.linspace(samples.min(), samples.max(), 500)
                pdf = np.zeros_like(x_plot)
                weights = np.array(var.weights) if hasattr(var.weights, '__len__') else [var.weights]
                means = np.array(var.means) if hasattr(var.means, '__len__') else [var.means]
                stdevs = np.array(var.stdevs) if hasattr(var.stdevs, '__len__') else [var.stdevs]
                
                for w, mu, sig in zip(weights, means, stdevs):
                    pdf += w * stats.norm.pdf(x_plot, mu, sig)
                axes[1, i//2].plot(x_plot, pdf, 'r-', lw=2, label='True PDF')
    
    axes[1, 0].set_xlabel('Value')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Variable Distributions')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Error distribution
    errors = np.abs(y_pce - y_mc) / (np.abs(y_mc) + 1e-10) * 100
    axes[1, 1].hist(errors, bins=50, edgecolor='black')
    axes[1, 1].set_xlabel('Relative Error (%)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title(f'Prediction Error (Mean: {np.mean(errors):.2f}%)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gmm_pce_verification.png', dpi=150)
    plt.show()
    
    # Summary
    mean_error = abs(pce_mean - mc_mean) / abs(mc_mean) * 100
    var_error = abs(pce_var - mc_var) / abs(mc_var) * 100
    
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    print(f"Mean relative error: {mean_error:.2f}%")
    print(f"Variance relative error: {var_error:.2f}%")
    
    if mean_error < 5 and var_error < 10:
        print("\n✓ VERIFICATION PASSED")
    else:
        print("\n✗ VERIFICATION FAILED")
    print("="*70)
    
    return mean_error < 5 and var_error < 10


if __name__ == '__main__':
    success = run_verification()
    exit(0 if success else 1)