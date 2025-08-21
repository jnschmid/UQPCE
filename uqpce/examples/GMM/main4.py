"""
GMM Orthogonal Polynomial Verification Script
Verifies the orthogonal polynomial construction for GMM variables in UQPCE
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.special import hermite
from sympy import symbols, lambdify, expand, simplify, exp as sym_exp, sqrt as sym_sqrt, pi as sym_pi
from sympy import integrate as sym_integrate
import os
from uqpce.mdao import interface


class GMM_OrthoPoly_Verifier:
    """Verify orthogonal polynomial construction for GMM variables"""
    
    def __init__(self, input_file, matrix_file):
        """Initialize with UQPCE model files"""
        (
            self.var_basis, self.norm_sq, self.resampled_var_basis,
            self.aleatory_cnt, self.epistemic_cnt, self.resp_cnt, 
            self.order, self.variables, self.sig, self.run_matrix
        ) = interface.initialize(input_file, matrix_file)
        
        # Find GMM variables
        self.gmm_vars = []
        self.gmm_indices = []
        for i, var in enumerate(self.variables):
            if hasattr(var, 'weights'):
                self.gmm_vars.append(var)
                self.gmm_indices.append(i)
    
    def verify_orthogonality(self, gmm_var, tol=1e-6):
        """Verify that the polynomials are orthogonal with respect to the GMM PDF"""
        print(f"\nVerifying Orthogonality for {gmm_var.name}:")
        print("="*60)
        
        # Get the orthogonal polynomials
        poly_vect = gmm_var.var_orthopoly_vect
        n_polys = len(poly_vect)
        
        # Create the PDF function for integration
        x = gmm_var.x  # Use the variable's symbolic x
        
        # Build GMM PDF symbolically using sympy
        pdf_expr = 0
        for w, mu, sigma in zip(gmm_var.weights, gmm_var.means, gmm_var.stdevs):
            # Use sympy functions for symbolic expression
            w_val = float(w)
            mu_val = float(mu)
            sigma_val = float(sigma)
            
            pdf_expr += (w_val / (sigma_val * sym_sqrt(2 * sym_pi))) * sym_exp(-((x - mu_val)**2) / (2 * sigma_val**2))
        
        # Convert to numerical function
        pdf_func = lambdify(x, pdf_expr, 'numpy')
        
        # Check orthogonality for first few polynomials
        max_check = min(n_polys, 5)  # Check first 5 polynomials
        orthogonality_matrix = np.zeros((max_check, max_check))
        
        print("\nOrthogonality Matrix (should be diagonal):")
        print("-" * 40)
        
        for i in range(max_check):
            for j in range(max_check):
                # Convert symbolic polynomials to functions
                poly_i = lambdify(x, poly_vect[i], 'numpy')
                poly_j = lambdify(x, poly_vect[j], 'numpy')
                
                # Define integrand
                def integrand(x_val):
                    try:
                        return poly_i(x_val) * poly_j(x_val) * pdf_func(x_val)
                    except:
                        return 0.0
                
                # Numerical integration
                try:
                    result, error = integrate.quad(
                        integrand, 
                        float(gmm_var.interval_low), 
                        float(gmm_var.interval_high),
                        limit=100,
                        epsabs=1e-10
                    )
                except:
                    result = 0.0
                    print(f"Warning: Integration failed for ({i},{j})")
                
                orthogonality_matrix[i, j] = result
        
        # Normalize to show structure better
        diagonal = np.diag(orthogonality_matrix).copy()  # Make a copy to avoid read-only issue
        # Avoid division by zero
        diagonal[diagonal == 0] = 1e-10
        norm_matrix = orthogonality_matrix / np.sqrt(np.outer(diagonal, diagonal))
        
        print("Normalized Inner Product Matrix:")
        print(np.array2string(norm_matrix, precision=4, suppress_small=True))
        
        # Print the raw orthogonality matrix too
        print("\nRaw Inner Product Matrix:")
        print(np.array2string(orthogonality_matrix, precision=6, suppress_small=True))
        
        # Check if matrix is diagonal
        off_diagonal = norm_matrix - np.diag(np.diag(norm_matrix))
        max_off_diagonal = np.max(np.abs(off_diagonal))
        
        print(f"\nMax off-diagonal element (normalized): {max_off_diagonal:.2e}")
        
        # Also check the condition number
        if np.linalg.det(orthogonality_matrix[:3, :3]) != 0:
            cond_number = np.linalg.cond(orthogonality_matrix[:3, :3])
            print(f"Condition number of first 3x3 submatrix: {cond_number:.2e}")
        
        if max_off_diagonal < tol:
            print("✅ Polynomials are orthogonal")
        else:
            print(f"⚠️  WARNING: Polynomials may not be orthogonal (tolerance: {tol})")
            print("   This could indicate issues with the recursive construction")
        
        return orthogonality_matrix, norm_matrix
    
    def verify_norm_squared(self, gmm_var):
        """Verify the norm squared values"""
        print(f"\nVerifying Norm Squared for {gmm_var.name}:")
        print("="*60)
        
        # Get stored norm squared values
        stored_norm_sq = gmm_var.norm_sq_vals
        
        # Calculate norm squared independently
        x = gmm_var.x
        
        # Build GMM PDF symbolically
        pdf_expr = 0
        for w, mu, sigma in zip(gmm_var.weights, gmm_var.means, gmm_var.stdevs):
            w_val = float(w)
            mu_val = float(mu)
            sigma_val = float(sigma)
            
            pdf_expr += (w_val / (sigma_val * sym_sqrt(2 * sym_pi))) * sym_exp(-((x - mu_val)**2) / (2 * sigma_val**2))
        
        pdf_func = lambdify(x, pdf_expr, 'numpy')
        
        calculated_norm_sq = []
        n_check = min(len(stored_norm_sq), 5)
        
        print("\nNorm Squared Comparison:")
        print("-" * 40)
        print("Order | Stored    | Calculated | Rel Error | Status")
        print("-" * 40)
        
        for i in range(n_check):
            poly = lambdify(x, gmm_var.var_orthopoly_vect[i], 'numpy')
            
            def integrand(x_val):
                try:
                    return poly(x_val)**2 * pdf_func(x_val)
                except:
                    return 0.0
            
            try:
                result, error = integrate.quad(
                    integrand,
                    float(gmm_var.interval_low),
                    float(gmm_var.interval_high),
                    limit=100,
                    epsabs=1e-10
                )
            except:
                result = 0.0
                print(f"Warning: Integration failed for norm squared {i}")
            
            calculated_norm_sq.append(result)
            
            if result != 0:
                rel_error = abs(stored_norm_sq[i] - result) / abs(result)
                status = "✓" if rel_error < 0.01 else "✗"
            else:
                rel_error = abs(stored_norm_sq[i])
                status = "✗"
            
            print(f"  {i:3d} | {stored_norm_sq[i]:9.6f} | {result:10.6f} | {rel_error:9.2e} | {status}")
        
        return stored_norm_sq[:n_check], calculated_norm_sq
    
    def check_pdf_integration(self, gmm_var):
        """Verify that the PDF integrates to 1"""
        print(f"\nVerifying PDF Integration for {gmm_var.name}:")
        print("="*60)
        
        x = gmm_var.x
        
        # Build GMM PDF
        pdf_expr = 0
        for w, mu, sigma in zip(gmm_var.weights, gmm_var.means, gmm_var.stdevs):
            w_val = float(w)
            mu_val = float(mu)
            sigma_val = float(sigma)
            
            pdf_expr += (w_val / (sigma_val * sym_sqrt(2 * sym_pi))) * sym_exp(-((x - mu_val)**2) / (2 * sigma_val**2))
        
        pdf_func = lambdify(x, pdf_expr, 'numpy')
        
        # Integrate PDF
        result, error = integrate.quad(
            pdf_func,
            float(gmm_var.interval_low),
            float(gmm_var.interval_high),
            limit=100
        )
        
        print(f"  PDF integral: {result:.6f} (should be ~1.0)")
        print(f"  Integration error: {error:.2e}")
        
        if abs(result - 1.0) > 0.01:
            print("  ⚠️  WARNING: PDF does not integrate to 1!")
            print("     This may indicate the interval bounds are too restrictive")
    
    def compare_with_hermite(self, gmm_var):
        """Compare GMM polynomials with standard Hermite polynomials"""
        print(f"\nComparing with Hermite Polynomials for {gmm_var.name}:")
        print("="*60)
        
        # For a single Gaussian, the orthogonal polynomials should be Hermite
        dominant_idx = np.argmax(gmm_var.weights)
        dominant_weight = gmm_var.weights[dominant_idx]
        
        print(f"Dominant component: {dominant_idx} with weight {dominant_weight:.3f}")
        
        # Always show the polynomial structure
        n_show = min(4, len(gmm_var.var_orthopoly_vect))
        print("\nFirst few polynomials:")
        for i in range(n_show):
            poly = gmm_var.var_orthopoly_vect[i]
            poly_expanded = expand(simplify(poly))
            print(f"  ψ_{i}: {poly_expanded}")
        
        if dominant_weight > 0.8:
            print("\nSingle component dominates - polynomials should resemble Hermite")
        else:
            print("\nMultiple significant components - polynomials will be complex")

    def plot_polynomials(self, gmm_var, n_plot=4):
        """Plot the orthogonal polynomials"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        # Create x range
        x_range = np.linspace(
            float(gmm_var.interval_low), 
            float(gmm_var.interval_high), 
            500
        )
        
        # Calculate GMM PDF for reference
        pdf_vals = np.zeros_like(x_range)
        for w, mu, sigma in zip(gmm_var.weights, gmm_var.means, gmm_var.stdevs):
            w_val = float(w)
            mu_val = float(mu)
            sigma_val = float(sigma)
            pdf_vals += w_val * np.exp(-(x_range - mu_val)**2 / (2*sigma_val**2)) / (sigma_val * np.sqrt(2*np.pi))
        
        for idx in range(min(n_plot, len(gmm_var.var_orthopoly_vect))):
            ax = axes[idx]
            
            # Plot polynomial
            poly_func = lambdify(gmm_var.x, gmm_var.var_orthopoly_vect[idx], 'numpy')
            
            try:
                poly_vals = poly_func(x_range)
                
                # Handle case where polynomial is constant
                if np.isscalar(poly_vals):
                    poly_vals = np.full_like(x_range, poly_vals)
                else:
                    poly_vals = np.array(poly_vals)
                    
                # Ensure it's the right shape
                if poly_vals.shape != x_range.shape:
                    if poly_vals.size == 1:
                        poly_vals = np.full_like(x_range, poly_vals.item())
                    else:
                        poly_vals = poly_vals.flatten()
                        
            except Exception as e:
                print(f"Warning: Could not evaluate polynomial {idx}: {e}")
                poly_vals = np.zeros_like(x_range)
            
            # Normalize for visualization
            if gmm_var.norm_sq_vals[idx] > 0:
                poly_vals_norm = poly_vals / np.sqrt(gmm_var.norm_sq_vals[idx])
            else:
                poly_vals_norm = poly_vals
            
            ax2 = ax.twinx()
            
            # Plot polynomial
            line1 = ax.plot(x_range, poly_vals_norm, 'b-', linewidth=2, label=f'$\\psi_{{{idx}}}(x)$')
            
            # Plot PDF
            line2 = ax2.plot(x_range, pdf_vals, 'r--', alpha=0.5, label='GMM PDF')
            ax2.fill_between(x_range, pdf_vals, alpha=0.2, color='red')
            
            # Mark component means
            for mu in gmm_var.means:
                ax.axvline(x=float(mu), color='g', linestyle=':', alpha=0.5, linewidth=1)
            
            ax.set_xlabel('x')
            ax.set_ylabel(f'$\\psi_{{{idx}}}(x)$ (normalized)', color='b')
            ax2.set_ylabel('PDF', color='r')
            ax.set_title(f'Orthogonal Polynomial {idx}')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='y', labelcolor='b')
            ax2.tick_params(axis='y', labelcolor='r')
            
            # Add zero line for reference
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
            
            # Add legend
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='best')
        
        plt.suptitle(f'Orthogonal Polynomials for GMM Variable: {gmm_var.name}')
        plt.tight_layout()
        plt.savefig('gmm_orthogonal_polynomials.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return fig

    def verify_recursion_relation(self, gmm_var):
        """Check if the recursive construction is working correctly"""
        print(f"\nVerifying Recursion Relation for {gmm_var.name}:")
        print("="*60)
        
        print("\nChecking polynomial degrees:")
        for i in range(min(5, len(gmm_var.var_orthopoly_vect))):
            poly = gmm_var.var_orthopoly_vect[i]
            poly_expanded = expand(poly)
            
            # Get the degree of the polynomial
            try:
                if poly_expanded != 0:
                    degree = poly_expanded.as_poly(gmm_var.x).degree()
                else:
                    degree = -1
            except:
                degree = -1
                print(f"  Warning: Could not determine degree for polynomial {i}")
            
            status = "✓" if degree == i else "✗"
            print(f"  Polynomial {i}: degree = {degree} (expected: {i}) {status}")
            
            if degree != i and degree != -1:
                print(f"    ⚠️  WARNING: Degree mismatch!")
    
    def verify_sampling_with_basis(self, gmm_var, n_samples=10000):
        """Verify that sampling respects the orthogonal basis"""
        print(f"\nVerifying Sampling with Orthogonal Basis for {gmm_var.name}:")
        print("="*60)
        
        # Generate samples
        samples = gmm_var.generate_samples(n_samples)
        
        # Also calculate the PDF-weighted expectation
        print("\nEmpirical moments vs theoretical (PDF-weighted):")
        print("-" * 40)
        print("Order | Empirical | Expected | Status")
        print("-" * 40)
        
        for i in range(min(5, len(gmm_var.var_orthopoly_vect))):
            poly_func = lambdify(gmm_var.x, gmm_var.var_orthopoly_vect[i], 'numpy')
            
            try:
                # Calculate empirical expectation
                empirical_moment = np.mean(poly_func(samples))
                
                if i == 0:
                    # First polynomial (constant) should have mean sqrt(norm_sq[0])
                    expected = np.sqrt(gmm_var.norm_sq_vals[0])
                    status = "✓" if abs(empirical_moment - expected) < 0.1 else "✗"
                    print(f"  {i:3d} | {empirical_moment:9.6f} | {expected:8.6f} | {status}")
                else:
                    # Higher order should be approximately 0
                    expected = 0.0
                    status = "✓" if abs(empirical_moment) < 0.1 else "✗"
                    print(f"  {i:3d} | {empirical_moment:9.6f} | {expected:8.6f} | {status}")
            except:
                print(f"  {i:3d} | Error evaluating polynomial")


def run_orthopoly_verification():
    """Run comprehensive orthogonal polynomial verification"""
    
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, 'input.yaml')
    matrix_file = os.path.join(script_dir, 'run_matrix.dat')
    
    # Initialize verifier
    verifier = GMM_OrthoPoly_Verifier(input_file, matrix_file)
    
    if len(verifier.gmm_vars) == 0:
        print("No GMM variables found!")
        return None
    
    print("\n" + "="*60)
    print("GMM ORTHOGONAL POLYNOMIAL VERIFICATION")
    print("="*60)
    
    results = {}
    
    for gmm_var in verifier.gmm_vars:
        print(f"\n{'='*60}")
        print(f"Analyzing Variable: {gmm_var.name}")
        print(f"{'='*60}")
        
        # Print basic info about the GMM
        print(f"\nGMM Components:")
        print("-" * 40)
        for i, (w, mu, sigma) in enumerate(zip(gmm_var.weights, gmm_var.means, gmm_var.stdevs)):
            print(f"  Component {i}: weight={w:.3f}, mean={mu:.3f}, stdev={sigma:.3f}")
        print(f"  Interval: [{gmm_var.interval_low}, {gmm_var.interval_high}]")
        
        # Check PDF integration first
        verifier.check_pdf_integration(gmm_var)
        
        # Run all verification tests
        ortho_matrix, norm_matrix = verifier.verify_orthogonality(gmm_var)
        stored_ns, calc_ns = verifier.verify_norm_squared(gmm_var)
        verifier.verify_recursion_relation(gmm_var)
        verifier.compare_with_hermite(gmm_var)
        verifier.verify_sampling_with_basis(gmm_var)
        
        # Plot polynomials
        fig = verifier.plot_polynomials(gmm_var)
        
        results[gmm_var.name] = {
            'orthogonality_matrix': ortho_matrix,
            'norm_squared_stored': stored_ns,
            'norm_squared_calculated': calc_ns
        }
    
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    # Summarize findings
    print("\nKey Findings:")
    for var_name in results.keys():
        ortho_mat = results[var_name]['orthogonality_matrix']
        off_diag = ortho_mat - np.diag(np.diag(ortho_mat))
        max_off = np.max(np.abs(off_diag))
        
        if max_off < 1e-6:
            print(f"  {var_name}: ✅ Orthogonal polynomials verified")
        else:
            print(f"  {var_name}: ⚠️  Potential orthogonality issues (max off-diag: {max_off:.2e})")
    
    return results


if __name__ == '__main__':
    results = run_orthopoly_verification()