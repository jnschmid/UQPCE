"""
GMM-PCE Verification Script
Verifies that the PCE surrogate correctly represents the underlying GMM distribution
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import gaussian_kde, ks_2samp, wasserstein_distance
import openmdao.api as om
from uqpce.mdao.uqpcegroup import UQPCEGroup
from uqpce.mdao import interface
import os


def gmm_pdf(x, weights, means, stdevs):
    """Calculate the true GMM PDF"""
    pdf = np.zeros_like(x)
    for w, mu, sigma in zip(weights, means, stdevs):
        pdf += w * stats.norm.pdf(x, mu, sigma)
    return pdf


def gmm_cdf(x, weights, means, stdevs):
    """Calculate the true GMM CDF"""
    cdf = np.zeros_like(x)
    for w, mu, sigma in zip(weights, means, stdevs):
        cdf += w * stats.norm.cdf(x, mu, sigma)
    return cdf


def gmm_samples(n, weights, means, stdevs):
    """Generate samples from the true GMM"""
    components = np.random.choice(len(weights), size=n, p=weights)
    samples = np.zeros(n)
    for i, (mu, sigma) in enumerate(zip(means, stdevs)):
        mask = components == i
        n_comp = np.sum(mask)
        if n_comp > 0:
            samples[mask] = np.random.normal(mu, sigma, n_comp)
    return samples


class GMM_PCE_Verifier:
    """Class to verify GMM representation in PCE surrogate"""
    
    def __init__(self, input_file, matrix_file):
        """Initialize with UQPCE model files"""
        # Load UQPCE model
        (
            self.var_basis, self.norm_sq, self.resampled_var_basis,
            self.aleatory_cnt, self.epistemic_cnt, self.resp_cnt, 
            self.order, self.variables, self.sig, self.run_matrix
        ) = interface.initialize(input_file, matrix_file)
        
        # Find GMM variables
        self.gmm_vars = []
        self.gmm_indices = []
        for i, var in enumerate(self.variables):
            if hasattr(var, 'weights'):  # It's a GMM variable
                self.gmm_vars.append(var)
                self.gmm_indices.append(i)
                
    def verify_sampling_distribution(self, n_samples=10000):
        """Verify that the GMM variable sampling follows the correct distribution"""
        results = {}
        
        for idx, gmm_var in zip(self.gmm_indices, self.gmm_vars):
            print(f"\nVerifying GMM Variable {gmm_var.name}:")
            print("="*60)
            
            # Generate samples from UQPCE implementation
            uqpce_samples = gmm_var.generate_samples(n_samples)
            
            # Generate samples from true GMM
            true_samples = gmm_samples(
                n_samples, gmm_var.weights, gmm_var.means, gmm_var.stdevs
            )
            
            # Statistical tests
            ks_stat, ks_pval = ks_2samp(uqpce_samples, true_samples)
            w_distance = wasserstein_distance(uqpce_samples, true_samples)
            
            # Moment comparison
            true_mean = np.sum(gmm_var.weights * gmm_var.means)
            true_var = np.sum(gmm_var.weights * (gmm_var.stdevs**2 + gmm_var.means**2)) - true_mean**2
            
            uqpce_mean = np.mean(uqpce_samples)
            uqpce_var = np.var(uqpce_samples)
            
            results[gmm_var.name] = {
                'ks_statistic': ks_stat,
                'ks_pvalue': ks_pval,
                'wasserstein_distance': w_distance,
                'mean_error': abs(uqpce_mean - true_mean),
                'variance_error': abs(uqpce_var - true_var),
                'true_mean': true_mean,
                'true_var': true_var,
                'empirical_mean': uqpce_mean,
                'empirical_var': uqpce_var
            }
            
            print(f"  KS Test p-value: {ks_pval:.4f} (>{0.05:.2f} is good)")
            print(f"  Wasserstein Distance: {w_distance:.6f}")
            print(f"  Mean - True: {true_mean:.4f}, Empirical: {uqpce_mean:.4f}, Error: {results[gmm_var.name]['mean_error']:.6f}")
            print(f"  Var  - True: {true_var:.4f}, Empirical: {uqpce_var:.4f}, Error: {results[gmm_var.name]['variance_error']:.6f}")
            
        return results
    
    def verify_pce_propagation(self, prob, n_verification=5000):
        """Verify PCE propagation preserves GMM characteristics"""
        print("\nVerifying PCE Propagation:")
        print("="*60)
        
        # Generate verification samples
        verification_samples = np.zeros((n_verification, len(self.variables)))
        for i, var in enumerate(self.variables):
            verification_samples[:, i] = var.generate_samples(n_verification)
        
        # Run through actual model
        true_responses = []
        for i in range(n_verification):
            prob.set_val('x', 1.0)  # Set design variable
            for j, var in enumerate(self.variables):
                prob.set_val(var.name, verification_samples[i, j])
            prob.run_model()
            true_responses.append(prob.get_val('f')[0])
        true_responses = np.array(true_responses)
        
        # Get PCE predictions
        from uqpce.pce import PCE
        pce_model = PCE()
        pce_model.variables = self.variables
        pce_model._var_count = len(self.variables)
        
        # Fit PCE with original run matrix and responses
        pce_model.fit(self.run_matrix, prob.get_val('f'))
        
        # Predict on verification samples
        pce_predictions = pce_model.predict(verification_samples)
        
        # Calculate errors
        abs_errors = np.abs(true_responses - pce_predictions.flatten())
        rel_errors = abs_errors / (np.abs(true_responses) + 1e-10)
        
        print(f"  Mean Absolute Error: {np.mean(abs_errors):.6f}")
        print(f"  Max Absolute Error: {np.max(abs_errors):.6f}")
        print(f"  Mean Relative Error: {np.mean(rel_errors)*100:.2f}%")
        print(f"  95th Percentile Error: {np.percentile(abs_errors, 95):.6f}")
        
        return {
            'mae': np.mean(abs_errors),
            'max_error': np.max(abs_errors),
            'mre': np.mean(rel_errors),
            'p95_error': np.percentile(abs_errors, 95)
        }
    
    def verify_moments_preservation(self, n_mc=50000):
        """Verify that moments are preserved through PCE"""
        print("\nVerifying Moment Preservation:")
        print("="*60)
        
        for idx, gmm_var in zip(self.gmm_indices, self.gmm_vars):
            # Theoretical moments
            theoretical_mean = np.sum(gmm_var.weights * gmm_var.means)
            theoretical_var = np.sum(gmm_var.weights * (gmm_var.stdevs**2 + gmm_var.means**2)) - theoretical_mean**2
            theoretical_skew = self._compute_skewness(gmm_var)
            theoretical_kurt = self._compute_kurtosis(gmm_var)
            
            # Monte Carlo moments from resampling
            mc_samples = gmm_var.generate_samples(n_mc)
            mc_mean = np.mean(mc_samples)
            mc_var = np.var(mc_samples)
            mc_skew = stats.skew(mc_samples)
            mc_kurt = stats.kurtosis(mc_samples)
            
            print(f"\n  Variable: {gmm_var.name}")
            print(f"    Mean      - Theory: {theoretical_mean:.4f}, MC: {mc_mean:.4f}, Error: {abs(theoretical_mean - mc_mean):.6f}")
            print(f"    Variance  - Theory: {theoretical_var:.4f}, MC: {mc_var:.4f}, Error: {abs(theoretical_var - mc_var):.6f}")
            print(f"    Skewness  - Theory: {theoretical_skew:.4f}, MC: {mc_skew:.4f}, Error: {abs(theoretical_skew - mc_skew):.6f}")
            print(f"    Kurtosis  - Theory: {theoretical_kurt:.4f}, MC: {mc_kurt:.4f}, Error: {abs(theoretical_kurt - mc_kurt):.6f}")
    
    def _compute_skewness(self, gmm_var):
        """Compute theoretical skewness of GMM"""
        mean = np.sum(gmm_var.weights * gmm_var.means)
        var = np.sum(gmm_var.weights * (gmm_var.stdevs**2 + gmm_var.means**2)) - mean**2
        
        third_moment = 0
        for w, mu, sigma in zip(gmm_var.weights, gmm_var.means, gmm_var.stdevs):
            # E[X^3] for each component
            third_moment += w * (mu**3 + 3*mu*sigma**2)
        
        central_third = third_moment - 3*mean*var - mean**3
        return central_third / (var**1.5)
    
    def _compute_kurtosis(self, gmm_var):
        """Compute theoretical excess kurtosis of GMM"""
        mean = np.sum(gmm_var.weights * gmm_var.means)
        var = np.sum(gmm_var.weights * (gmm_var.stdevs**2 + gmm_var.means**2)) - mean**2
        
        # Calculate third moment (needed for fourth central moment)
        third_moment = 0
        for w, mu, sigma in zip(gmm_var.weights, gmm_var.means, gmm_var.stdevs):
            # E[X^3] for each component
            third_moment += w * (mu**3 + 3*mu*sigma**2)
        
        # Calculate fourth moment
        fourth_moment = 0
        for w, mu, sigma in zip(gmm_var.weights, gmm_var.means, gmm_var.stdevs):
            # E[X^4] for each component
            fourth_moment += w * (mu**4 + 6*mu**2*sigma**2 + 3*sigma**4)
        
        # Calculate fourth central moment
        central_fourth = fourth_moment - 4*mean*third_moment + 6*mean**2*var + mean**4
        
        # Return excess kurtosis
        return central_fourth / (var**2) - 3
    
    def plot_verification(self, n_samples=10000):
        """Create visualization of GMM representation"""
        
        if len(self.gmm_vars) == 0:
            print("No GMM variables found to plot")
            return None
            
        fig, axes = plt.subplots(len(self.gmm_vars), 3, figsize=(15, 5*len(self.gmm_vars)))
        if len(self.gmm_vars) == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (gmm_idx, gmm_var) in enumerate(zip(self.gmm_indices, self.gmm_vars)):
            # Generate samples
            uqpce_samples = gmm_var.generate_samples(n_samples)
            true_samples = gmm_samples(n_samples, gmm_var.weights, gmm_var.means, gmm_var.stdevs)
            
            # PDF comparison
            x_range = np.linspace(gmm_var.interval_low, gmm_var.interval_high, 1000)
            true_pdf = gmm_pdf(x_range, gmm_var.weights, gmm_var.means, gmm_var.stdevs)
            
            axes[idx, 0].plot(x_range, true_pdf, 'r-', label='True GMM', linewidth=2)
            axes[idx, 0].hist(uqpce_samples, bins=50, density=True, alpha=0.5, label='UQPCE Samples')
            axes[idx, 0].set_title(f'{gmm_var.name}: PDF Comparison')
            axes[idx, 0].set_xlabel('Value')
            axes[idx, 0].set_ylabel('Density')
            axes[idx, 0].legend()
            axes[idx, 0].grid(True, alpha=0.3)
            
            # CDF comparison
            true_cdf = gmm_cdf(x_range, gmm_var.weights, gmm_var.means, gmm_var.stdevs)
            empirical_cdf = np.array([np.mean(uqpce_samples <= x) for x in x_range])
            
            axes[idx, 1].plot(x_range, true_cdf, 'r-', label='True GMM', linewidth=2)
            axes[idx, 1].plot(x_range, empirical_cdf, 'b--', label='UQPCE Empirical', linewidth=2)
            axes[idx, 1].set_title(f'{gmm_var.name}: CDF Comparison')
            axes[idx, 1].set_xlabel('Value')
            axes[idx, 1].set_ylabel('Cumulative Probability')
            axes[idx, 1].legend()
            axes[idx, 1].grid(True, alpha=0.3)
            
            # Q-Q plot
            q_true = np.percentile(true_samples, np.linspace(0, 100, 100))
            q_uqpce = np.percentile(uqpce_samples, np.linspace(0, 100, 100))
            
            axes[idx, 2].scatter(q_true, q_uqpce, alpha=0.5)
            
            # Add reference line
            min_val = min(q_true.min(), q_uqpce.min())
            max_val = max(q_true.max(), q_uqpce.max())
            axes[idx, 2].plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')
            
            axes[idx, 2].set_title(f'{gmm_var.name}: Q-Q Plot')
            axes[idx, 2].set_xlabel('True GMM Quantiles')
            axes[idx, 2].set_ylabel('UQPCE Quantiles')
            axes[idx, 2].legend()
            axes[idx, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('gmm_verification.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def verify_standardization(self, n_samples=1000):
        """Verify that standardization/unstandardization works correctly"""
        print("\nVerifying Standardization/Unstandardization:")
        print("="*60)
        
        for gmm_var in self.gmm_vars:
            # Generate test points
            test_points = gmm_var.generate_samples(n_samples)
            
            # Standardize
            standardized = gmm_var.standardize_points(test_points)
            
            # Unstandardize
            recovered = gmm_var.unstandardize_points(standardized)
            
            # Check round-trip error
            round_trip_error = np.max(np.abs(test_points - recovered))
            
            print(f"\n  Variable: {gmm_var.name}")
            print(f"    Original range: [{test_points.min():.4f}, {test_points.max():.4f}]")
            print(f"    Standardized range: [{standardized.min():.4f}, {standardized.max():.4f}]")
            print(f"    Round-trip error: {round_trip_error:.2e}")
            
            if round_trip_error > 1e-10:
                print(f"    ⚠️  WARNING: Large round-trip error detected!")


def run_comprehensive_verification():
    """Run full verification suite"""
    
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, 'input.yaml')
    matrix_file = os.path.join(script_dir, 'run_matrix.dat')
    
    # Initialize verifier
    verifier = GMM_PCE_Verifier(input_file, matrix_file)
    
    # Run verification tests
    print("\n" + "="*60)
    print("GMM-PCE VERIFICATION REPORT")
    print("="*60)
    
    # Test 1: Sampling distribution
    sampling_results = verifier.verify_sampling_distribution(n_samples=50000)
    
    # Test 2: Moment preservation
    verifier.verify_moments_preservation(n_mc=100000)
    
    # Test 3: Standardization verification
    verifier.verify_standardization(n_samples=10000)
    
    # Test 4: Visual verification
    fig = verifier.plot_verification(n_samples=20000)
    
    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    all_passed = True
    for var_name, results in sampling_results.items():
        if results['ks_pvalue'] < 0.01:
            print(f"⚠️  WARNING: {var_name} may have distribution issues (KS p-value: {results['ks_pvalue']:.4f})")
            all_passed = False
        if results['mean_error'] > 0.1:
            print(f"⚠️  WARNING: {var_name} has large mean error: {results['mean_error']:.4f}")
            all_passed = False
        if results['variance_error'] > 0.2:
            print(f"⚠️  WARNING: {var_name} has large variance error: {results['variance_error']:.4f}")
            all_passed = False
    
    if all_passed:
        print("✅ All GMM variables passed verification tests!")
    else:
        print("⚠️  Some issues detected - review warnings above")
    
    return sampling_results, fig


if __name__ == '__main__':
    results, fig = run_comprehensive_verification()