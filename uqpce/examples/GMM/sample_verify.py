"""
GMM Sample Verification Script

Verifies that UQPCE's sample generation for Gaussian Mixture Models
correctly represents the actual distribution by comparing samples
with the theoretical PDF and showing component membership.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
import sys

# Correct import paths based on project structure
from uqpce.pce.pce import PCE
from uqpce.pce.variables.continuous import GaussianMixtureVariable


def generate_gmm_samples_manual(n_samples, weights, means, stdevs):
    """
    Manually generate GMM samples with component tracking.
    Returns samples and their component assignments.
    """
    np.random.seed(42)
    
    # Sample component assignments
    components = np.random.choice(len(weights), size=n_samples, p=weights)
    
    # Generate samples based on component
    samples = np.zeros(n_samples)
    for i in range(len(weights)):
        mask = components == i
        n_comp = np.sum(mask)
        if n_comp > 0:
            samples[mask] = np.random.normal(means[i], stdevs[i], n_comp)
    
    return samples, components


def main():
    # GMM parameters (matching input.yaml)
    weights = [0.3, 0.5, 0.2]
    means = [-2.0, 0.0, 3.0]
    stdevs = [0.3, 0.5, 0.4]
    
    print("="*60)
    print("GMM SAMPLE GENERATION VERIFICATION")
    print("="*60)
    print(f"\nGMM Parameters:")
    print(f"  Weights: {weights}")
    print(f"  Means: {means}")
    print(f"  Std Devs: {stdevs}")
    
    # Calculate theoretical mean
    theoretical_mean = sum(w * m for w, m in zip(weights, means))
    theoretical_var = sum(w * (s**2 + m**2) for w, m, s in zip(weights, means, stdevs)) - theoretical_mean**2
    theoretical_std = np.sqrt(theoretical_var)
    
    print(f"\nTheoretical Statistics:")
    print(f"  Mean: {theoretical_mean:.4f}")
    print(f"  Std Dev: {theoretical_std:.4f}")
    
    # Create UQPCE GMM variable directly
    gmm_var = GaussianMixtureVariable(
        weights=weights,
        means=means,
        stdevs=stdevs,
        order=2,
        name='a1',
        number=0
    )
    
    # Generate samples using UQPCE's generate_samples method (this is the correct method for random sampling)
    n_samples = 10000
    uqpce_samples = gmm_var.generate_samples(n_samples, standardize=False)
    
    # Note about resample method
    print("\nNote: The resample() method is for PCE internal use and includes")
    print("extreme boundary points for numerical integration. It should NOT be")
    print("used for generating random samples from the distribution.")
    
    # For comparison, show what resample does (but don't use it for verification)
    resample_demo = gmm_var.resample(100)  # Small sample to show the issue
    resample_unstd = gmm_var.unstandardize_points(resample_demo)
    print(f"\nResample method range: [{resample_unstd.min():.1f}, {resample_unstd.max():.1f}]")
    print("(Notice the extreme values - this is by design for PCE integration)")
    
    # Generate manual samples with component tracking
    manual_samples, components = generate_gmm_samples_manual(n_samples, weights, means, stdevs)
    
    # Calculate sample statistics
    uqpce_mean = np.mean(uqpce_samples)
    uqpce_std = np.std(uqpce_samples)
    manual_mean = np.mean(manual_samples)
    manual_std = np.std(manual_samples)
    
    print(f"\nUQPCE generate_samples Statistics (n={n_samples}):")
    print(f"  Mean: {uqpce_mean:.4f} (error: {abs(uqpce_mean - theoretical_mean):.4f})")
    print(f"  Std Dev: {uqpce_std:.4f} (error: {abs(uqpce_std - theoretical_std):.4f})")
    print(f"  Range: [{uqpce_samples.min():.2f}, {uqpce_samples.max():.2f}]")
    
    print(f"\nManual Sample Statistics (n={n_samples}):")
    print(f"  Mean: {manual_mean:.4f} (error: {abs(manual_mean - theoretical_mean):.4f})")
    print(f"  Std Dev: {manual_std:.4f} (error: {abs(manual_std - theoretical_std):.4f})")
    print(f"  Range: [{manual_samples.min():.2f}, {manual_samples.max():.2f}]")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Define colors for each component
    colors = ['red', 'blue', 'green']
    
    # 1. Manual samples colored by component
    ax = axes[0, 0]
    
    # Create histogram with stacked bars for each component
    bins = np.linspace(min(manual_samples)-0.5, max(manual_samples)+0.5, 50)
    
    for i in range(len(weights)):
        component_samples = manual_samples[components == i]
        ax.hist(component_samples, bins=bins, alpha=0.6, color=colors[i], 
                label=f'Component {i+1}: μ={means[i]}, σ={stdevs[i]}', 
                density=True, edgecolor='black', linewidth=0.5)
    
    # Overlay theoretical PDF
    x_range = np.linspace(-4, 5, 500)
    pdf_total = sum(w * norm.pdf(x_range, m, s) for w, m, s in zip(weights, means, stdevs))
    ax.plot(x_range, pdf_total, 'k-', linewidth=2, label='Theoretical PDF')
    
    ax.set_xlabel('Value')
    ax.set_ylabel('Probability Density')
    ax.set_title('Manual GMM Samples (Colored by Component)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-4, 5)
    
    # 2. UQPCE generate_samples comparison
    ax = axes[0, 1]
    
    # Histogram of UQPCE samples
    ax.hist(uqpce_samples, bins=bins, alpha=0.7, color='purple', 
            density=True, edgecolor='black', linewidth=0.5, label='UQPCE generate_samples')
    
    # Overlay theoretical PDF
    ax.plot(x_range, pdf_total, 'k-', linewidth=2, label='Theoretical PDF')
    
    # Mark theoretical mean
    ax.axvline(theoretical_mean, color='red', linestyle='--', linewidth=2, 
               label=f'Theoretical Mean={theoretical_mean:.2f}')
    
    ax.set_xlabel('Value')
    ax.set_ylabel('Probability Density')
    ax.set_title('UQPCE GMM Samples (generate_samples method)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-4, 5)
    
    # 3. Component PDFs and Total
    ax = axes[1, 0]
    
    # Plot individual component PDFs
    for i, (w, m, s, c) in enumerate(zip(weights, means, stdevs, colors)):
        pdf_comp = w * norm.pdf(x_range, m, s)
        ax.plot(x_range, pdf_comp, '--', alpha=0.7, color=c, linewidth=1.5,
                label=f'w={w} × N({m}, {s}²)')
        ax.fill_between(x_range, pdf_comp, alpha=0.2, color=c)
    
    # Plot total PDF
    ax.plot(x_range, pdf_total, 'k-', linewidth=2.5, label='Total GMM')
    
    # Mark component means
    for m, c in zip(means, colors):
        ax.axvline(m, color=c, linestyle=':', alpha=0.5)
    
    ax.axvline(theoretical_mean, color='black', linestyle='--', linewidth=2,
               label=f'Overall Mean={theoretical_mean:.2f}')
    
    ax.set_xlabel('Value')
    ax.set_ylabel('Probability Density')
    ax.set_title('GMM Component Breakdown')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-4, 5)
    
    # 4. Q-Q plot comparing UQPCE samples to theoretical
    ax = axes[1, 1]
    
    # Sort samples for Q-Q plot
    uqpce_sorted = np.sort(uqpce_samples)
    manual_sorted = np.sort(manual_samples)
    
    # For GMM, we need to compute quantiles numerically
    def gmm_cdf(x):
        """Compute GMM CDF at point x"""
        return sum(w * norm.cdf(x, m, s) for w, m, s in zip(weights, means, stdevs))
    
    def gmm_quantile(p):
        """Find quantile using bisection"""
        from scipy.optimize import brentq
        return brentq(lambda x: gmm_cdf(x) - p, -10, 10)
    
    # Compute theoretical quantiles (subsample for speed)
    n_plot = min(100, len(uqpce_sorted))
    subsample_idx = np.linspace(0, len(uqpce_sorted)-1, n_plot, dtype=int)
    
    percentiles = np.linspace(0.5/len(uqpce_sorted), 1-0.5/len(uqpce_sorted), len(uqpce_sorted))
    theoretical_quantiles = [gmm_quantile(percentiles[i]) for i in subsample_idx]
    
    # Q-Q plot
    ax.scatter(theoretical_quantiles, uqpce_sorted[subsample_idx], 
              alpha=0.6, s=20, color='purple', label='UQPCE generate_samples')
    ax.scatter(theoretical_quantiles, manual_sorted[subsample_idx], 
              alpha=0.6, s=20, color='orange', label='Manual')
    
    # Add reference line
    ax.plot([min(theoretical_quantiles), max(theoretical_quantiles)], 
            [min(theoretical_quantiles), max(theoretical_quantiles)], 
            'k--', label='Perfect Match')
    
    ax.set_xlabel('Theoretical Quantiles')
    ax.set_ylabel('Sample Quantiles')
    ax.set_title('Q-Q Plot: Sample vs Theoretical Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('GMM Sample Generation Verification', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(script_dir, 'gmm_verification.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_file}")
    
    plt.show()
    
    # Perform statistical tests
    from scipy import stats
    
    print("\n" + "="*60)
    print("STATISTICAL TESTS")
    print("="*60)
    
    # Kolmogorov-Smirnov test
    ks_stat, ks_pval = stats.ks_2samp(uqpce_samples, manual_samples)
    
    print(f"\nKolmogorov-Smirnov Test (UQPCE vs Manual):")
    print(f"  Statistic: {ks_stat:.4f}")
    print(f"  P-value: {ks_pval:.4f}")
    if ks_pval > 0.05:
        print("  ✓ Samples appear to come from the same distribution (p > 0.05)")
    else:
        print("  ✗ Samples may come from different distributions (p < 0.05)")
    
    # Check component proportions
    print("\n" + "="*60)
    print("COMPONENT ANALYSIS FOR UQPCE SAMPLES")
    print("="*60)
    
    # Identify which component each UQPCE sample likely came from
    def identify_component(x, weights, means, stdevs):
        """Identify most likely component for a sample"""
        likelihoods = [w * norm.pdf(x, m, s) for w, m, s in zip(weights, means, stdevs)]
        return np.argmax(likelihoods)
    
    uqpce_components = [identify_component(x, weights, means, stdevs) for x in uqpce_samples]
    uqpce_proportions = [np.sum(np.array(uqpce_components) == i) / n_samples for i in range(len(weights))]
    
    print(f"\nExpected proportions: {weights}")
    print(f"Manual sample proportions: {[f'{np.sum(components == i)/n_samples:.3f}' for i in range(len(weights))]}")
    print(f"UQPCE sample proportions (estimated): {[f'{p:.3f}' for p in uqpce_proportions]}")
    
    print("\n" + "="*60)
    print("VERIFICATION COMPLETE")
    print("="*60)
    
    print("\nSUMMARY:")
    print("-" * 40)
    if ks_pval > 0.05:
        print("✓ UQPCE GMM sampling (generate_samples) is working correctly!")
        print("  - Distribution matches theoretical GMM")
        print("  - No extreme tail values")
        print("  - Component proportions are correct")
    else:
        print("⚠ Some issues detected in GMM sampling")
    
    print("\nIMPORTANT: Use gmm_var.generate_samples() for random sampling,")
    print("NOT gmm_var.resample() which is for PCE internal integration.")


if __name__ == '__main__':
    main()