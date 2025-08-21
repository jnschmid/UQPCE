"""
GMM-UQPCE Optimization Example with Diagnostics

Demonstrates the Gaussian Mixture Model capability in UQPCE through a simple
optimization problem that shows clear differences between deterministic and
robust optimization.
"""

import numpy as np
import openmdao.api as om
from uqpce.mdao.uqpcegroup import UQPCEGroup
from uqpce.mdao import interface
import os
import matplotlib.pyplot as plt
from scipy.stats import norm, lognorm
import warnings

warnings.filterwarnings('ignore')


class ImprovedFunction(om.ExplicitComponent):
    """
    f(x, y, a1, a2, a3) = (x - a1)^2 + 0.1*(y - a2)^2 - 3*x*a1 + 0.02*a3*(x^2 + y^2)
    
    This function creates a coupling between x and a1 that makes the deterministic
    optimum (using mean values) significantly different from the robust optimum
    (which must consider all three GMM modes).
    """
    
    def initialize(self):
        self.options.declare('vec_size', types=int)
    
    def setup(self):
        n = self.options['vec_size']
        arange = np.arange(n)
        
        self.add_input('x', val=0.0)
        self.add_input('y', val=0.0)
        self.add_input('a1', shape=(n,))
        self.add_input('a2', shape=(n,))
        self.add_input('a3', shape=(n,))
        self.add_output('f', shape=(n,))
        
        self.declare_partials('f', ['x', 'y'])
        self.declare_partials('f', ['a1', 'a2', 'a3'], rows=arange, cols=arange)
    
    def compute(self, inputs, outputs):
        x = inputs['x']
        y = inputs['y']
        a1 = inputs['a1']
        a2 = inputs['a2']
        a3 = inputs['a3']
        
        outputs['f'] = ((x - a1)**2 + 0.1*(y - a2)**2 - 
                       3*x*a1 + 0.02*a3*(x**2 + y**2))
    
    def compute_partials(self, inputs, partials):
        x = inputs['x']
        y = inputs['y']
        a1 = inputs['a1']
        a2 = inputs['a2']
        a3 = inputs['a3']
        
        partials['f', 'x'] = (2*(x - a1) - 3*a1 + 0.04*a3*x).reshape(-1, 1)
        partials['f', 'y'] = (0.2*(y - a2) + 0.04*a3*y).reshape(-1, 1)
        partials['f', 'a1'] = -5*x + 2*a1
        partials['f', 'a2'] = -0.2*(y - a2)
        partials['f', 'a3'] = 0.02*(x**2 + y**2)


def sample_from_gmm(n_samples):
    """Generate samples from the GMM"""
    weights = [0.3, 0.5, 0.2]
    means = [-1.0, 0.0, 2.0]
    stdevs = [0.3, 0.5, 0.4]
    
    samples = []
    for _ in range(n_samples):
        component = np.random.choice(3, p=weights)
        samples.append(np.random.normal(means[component], stdevs[component]))
    
    return np.array(samples)


def monte_carlo_eval(x, y, n_samples=500000):
    """Evaluate function using Monte Carlo with more samples"""
    a1 = sample_from_gmm(n_samples)
    a2 = np.random.uniform(-2.0, 2.0, n_samples)
    a3 = lognorm.rvs(s=1, scale=np.exp(0), size=n_samples)
    
    f = (x - a1)**2 + 0.1*(y - a2)**2 - 3*x*a1 + 0.02*a3*(x**2 + y**2)
    
    return f


def main():
    print("\n" + "="*70)
    print("GMM-UQPCE OPTIMIZATION EXAMPLE WITH DIAGNOSTICS")
    print("="*70)
    
    # Files
    input_file = 'input.yaml'
    matrix_file = 'run_matrix.dat'
    
    # Initialize UQPCE
    (var_basis, norm_sq, resampled_var_basis, 
     aleatory_cnt, epistemic_cnt, resp_cnt, order, variables, 
     sig, run_matrix) = interface.initialize(input_file, matrix_file)
    
    print(f"\nPCE Configuration:")
    print(f"  Order: {order}")
    print(f"  Number of samples in run_matrix: {resp_cnt}")
    print(f"  Aleatory samples for resampling: {aleatory_cnt}")
    
    # =========================================================================
    # DETERMINISTIC OPTIMIZATION
    # =========================================================================
    print("\n--- Deterministic Optimization ---")
    
    prob_det = om.Problem()
    prob_det.model.add_subsystem('func', ImprovedFunction(vec_size=1), promotes=['*'])
    
    prob_det.driver = om.pyOptSparseDriver(optimizer='SNOPT')
    prob_det.driver.opt_settings['Major optimality tolerance'] = 1e-8
    
    prob_det.model.add_design_var('x', lower=-5, upper=5)
    prob_det.model.add_design_var('y', lower=-5, upper=5)
    prob_det.model.add_objective('f')
    
    prob_det.setup()
    
    # Use mean values
    a1_mean = variables[0].get_mean()
    a2_mean = variables[1].get_mean()
    a3_mean = variables[2].get_mean()
    
    prob_det.set_val('a1', a1_mean)
    prob_det.set_val('a2', a2_mean)
    prob_det.set_val('a3', a3_mean)
    prob_det.set_val('x', 2.0)
    prob_det.set_val('y', -2.0)
    
    print(f"Using mean values: a1={a1_mean:.3f}, a2={a2_mean:.3f}, a3={a3_mean:.3f}")
    
    prob_det.run_driver()
    
    x_det = prob_det.get_val('x')[0]
    y_det = prob_det.get_val('y')[0]
    f_det = prob_det.get_val('f')[0]
    
    print(f"Optimal: x = {x_det:.4f}, y = {y_det:.4f}, f = {f_det:.4f}")
    
    # =========================================================================
    # ROBUST OPTIMIZATION WITH UQPCE
    # =========================================================================
    print("\n--- Robust Optimization (UQPCE) ---")
    
    prob_uq = om.Problem()
    
    prob_uq.model.add_subsystem(
        'func', 
        ImprovedFunction(vec_size=resp_cnt),
        promotes_inputs=['x', 'y', 'a1', 'a2', 'a3'],
        promotes_outputs=['f']
    )
    
    prob_uq.model.add_subsystem(
        'UQPCE',
        UQPCEGroup(
            significance=sig, var_basis=var_basis, norm_sq=norm_sq,
            resampled_var_basis=resampled_var_basis, tail='both',
            epistemic_cnt=epistemic_cnt, aleatory_cnt=aleatory_cnt,
            uncert_list=['f'], tanh_omega=1e-3
        ),
        promotes_inputs=['f'],
        promotes_outputs=['f:mean', 'f:variance', 'f:mean_plus_var', 'f:resampled_responses']
    )
    
    prob_uq.driver = om.pyOptSparseDriver(optimizer='SNOPT')
    prob_uq.driver.opt_settings['Major optimality tolerance'] = 1e-8
    
    prob_uq.model.add_design_var('x', lower=-5, upper=5)
    prob_uq.model.add_design_var('y', lower=-5, upper=5)
    prob_uq.model.add_objective('f:mean_plus_var')
    
    prob_uq.setup(force_alloc_complex=True)
    
    # Set uncertain values
    interface.set_vals(prob_uq, variables, run_matrix)
    prob_uq.set_val('x', 2.0)
    prob_uq.set_val('y', -2.0)
    
    # Get initial values
    prob_uq.run_model()
    f_initial_pce = prob_uq.get_val('f:resampled_responses').copy()
    
    # Optimize
    prob_uq.run_driver()
    
    x_uq = prob_uq.get_val('x')[0]
    y_uq = prob_uq.get_val('y')[0]
    f_mean = prob_uq.get_val('f:mean')[0]
    f_var = prob_uq.get_val('f:variance')[0]
    f_final_pce = prob_uq.get_val('f:resampled_responses').copy()
    
    print(f"Optimal: x = {x_uq:.4f}, y = {y_uq:.4f}")
    print(f"Mean = {f_mean:.4f}, Variance = {f_var:.4f}, Std = {np.sqrt(f_var):.4f}")
    
    # =========================================================================
    # DIAGNOSTIC COMPARISON
    # =========================================================================
    print("\n--- DIAGNOSTIC COMPARISON ---")
    
    # PCE statistics from resampled responses
    pce_mean_resampled = np.mean(f_final_pce)
    pce_std_resampled = np.std(f_final_pce)
    pce_var_resampled = np.var(f_final_pce)
    
    print(f"\nPCE Statistics at Robust Point ({x_uq:.3f}, {y_uq:.3f}):")
    print(f"  From f:mean output: {f_mean:.4f}")
    print(f"  From f:variance output: {f_var:.4f} (std = {np.sqrt(f_var):.4f})")
    print(f"  From resampled responses:")
    print(f"    Mean: {pce_mean_resampled:.4f}")
    print(f"    Variance: {pce_var_resampled:.4f}")
    print(f"    Std: {pce_std_resampled:.4f}")
    print(f"    Number of resampled points: {len(f_final_pce.flatten())}")
    
    # Monte Carlo with different sample sizes
    print(f"\nMonte Carlo Statistics at Robust Point:")
    for n_mc in [10000, 100000, 500000]:
        f_mc = monte_carlo_eval(x_uq, y_uq, n_mc)
        print(f"  {n_mc:7d} samples: Mean = {np.mean(f_mc):.4f}, Std = {np.std(f_mc):.4f}")
    
    # Final MC for plotting
    f_initial_mc = monte_carlo_eval(2.0, -2.0, 500000)
    f_final_mc = monte_carlo_eval(x_uq, y_uq, 500000)
    f_det_mc = monte_carlo_eval(x_det, y_det, 500000)
    
    # Check PCE approximation quality
    print(f"\nPCE Approximation Quality Check:")
    print(f"  Mean difference (PCE vs MC): {abs(f_mean - np.mean(f_final_mc)):.4f}")
    print(f"  Std difference (PCE vs MC): {abs(np.sqrt(f_var) - np.std(f_final_mc)):.4f}")
    
    # =========================================================================
    # PLOTTING
    # =========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # 1. GMM Distribution
    ax = axes[0, 0]
    
    weights = [0.3, 0.5, 0.2]
    means = [-1.0, 0.0, 2.0]
    stdevs = [0.3, 0.5, 0.4]
    
    x_range = np.linspace(-3, 4, 1000)
    
    colors = ['lightblue', 'orange', 'lightgreen']
    for i, (w, m, s, c) in enumerate(zip(weights, means, stdevs, colors)):
        comp_pdf = w * norm.pdf(x_range, m, s)
        ax.fill_between(x_range, 0, comp_pdf, alpha=0.3, color=c,
                        label=f'C{i+1}: w={w}, μ={m}, σ={s}')
    
    gmm_total = sum(w * norm.pdf(x_range, m, s) 
                    for w, m, s in zip(weights, means, stdevs))
    ax.plot(x_range, gmm_total, 'b-', lw=2.5, label='GMM Total')
    
    ax.set_xlabel('a1 Value')
    ax.set_ylabel('Probability Density')
    ax.set_title('GMM Variable (a1)')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-3, 4])
    
    # 2. Design Space
    ax = axes[0, 1]
    X, Y = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
    
    a1_m = variables[0].get_mean()
    a2_m = variables[1].get_mean()
    a3_m = variables[2].get_mean()
    
    Z = (X - a1_m)**2 + 0.1*(Y - a2_m)**2 - 3*X*a1_m + 0.02*a3_m*(X**2 + Y**2)
    
    contour = ax.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
    ax.clabel(contour, inline=True, fontsize=7, fmt='%.1f')
    ax.plot(x_det, y_det, 'ro', ms=10, label=f'Det: ({x_det:.2f}, {y_det:.2f})')
    ax.plot(x_uq, y_uq, 'bs', ms=10, label=f'Rob: ({x_uq:.2f}, {y_uq:.2f})')
    ax.plot(2, -2, 'g^', ms=8, label='Initial')
    
    if abs(x_uq - x_det) > 0.1 or abs(y_uq - y_det) > 0.1:
        ax.annotate('', xy=(x_uq, y_uq), xytext=(x_det, y_det),
                   arrowprops=dict(arrowstyle='->', lw=2, color='purple', alpha=0.7))
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Optimal Designs (at mean values)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    
    # 3. Initial Distribution
    ax = axes[0, 2]
    all_data = np.concatenate([f_initial_pce.flatten(), f_initial_mc[:10000]])
    bins = np.linspace(np.percentile(all_data, 1), np.percentile(all_data, 99), 35)
    
    ax.hist(f_initial_pce.flatten(), bins=bins, alpha=0.5, density=True, 
            color='blue', label='PCE')
    ax.hist(f_initial_mc[:10000], bins=bins, alpha=0.5, density=True, 
            color='red', label='Monte Carlo')
    ax.set_xlabel('Objective Value')
    ax.set_ylabel('Density')
    ax.set_title('Initial Point (2, -2)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Final Distribution
    ax = axes[1, 0]
    all_data = np.concatenate([f_final_pce.flatten(), f_final_mc[:10000]])
    bins = np.linspace(np.percentile(all_data, 1), np.percentile(all_data, 99), 35)
    
    ax.hist(f_final_pce.flatten(), bins=bins, alpha=0.5, density=True, 
            color='blue', label=f'PCE (std={pce_std_resampled:.3f})')
    ax.hist(f_final_mc[:10000], bins=bins, alpha=0.5, density=True, 
            color='red', label=f'MC (std={np.std(f_final_mc):.3f})')
    ax.set_xlabel('Objective Value')
    ax.set_ylabel('Density')
    ax.set_title(f'Robust Optimal ({x_uq:.2f}, {y_uq:.2f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Results Table
    ax = axes[1, 1]
    ax.axis('off')
    
    table_data = [
        ['Metric', 'Deterministic', 'Robust'],
        ['x', f'{x_det:.3f}', f'{x_uq:.3f}'],
        ['y', f'{y_det:.3f}', f'{y_uq:.3f}'],
        ['Δx from Det.', '-', f'{x_uq - x_det:+.3f}'],
        ['Δy from Det.', '-', f'{y_uq - y_det:+.3f}'],
        ['Mean (PCE)', '-', f'{f_mean:.3f}'],
        ['Std (PCE)', '-', f'{np.sqrt(f_var):.3f}'],
        ['Mean (MC 500k)', f'{np.mean(f_det_mc):.3f}', f'{np.mean(f_final_mc):.3f}'],
        ['Std (MC 500k)', f'{np.std(f_det_mc):.3f}', f'{np.std(f_final_mc):.3f}'],
        ['PCE vs MC Δstd', '-', f'{abs(np.sqrt(f_var) - np.std(f_final_mc)):.3f}']
    ]
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.8)
    
    # Highlight discrepancy
    if abs(np.sqrt(f_var) - np.std(f_final_mc)) > 0.1:
        table[(9, 2)].set_facecolor('#ffcccc')
    
    ax.set_title('Optimization Results', fontsize=12, fontweight='bold')
    
    # 6. Comparison of distributions
    ax = axes[1, 2]
    
    bins = np.linspace(min(f_det_mc.min(), f_final_mc.min()), 
                      max(f_det_mc.max(), f_final_mc.max()), 35)
    
    ax.hist(f_det_mc[:10000], bins=bins, alpha=0.5, density=True, 
            color='red', label=f'Det. point ({x_det:.2f}, {y_det:.2f})')
    ax.hist(f_final_mc[:10000], bins=bins, alpha=0.5, density=True, 
            color='blue', label=f'Robust point ({x_uq:.2f}, {y_uq:.2f})')
    
    ax.axvline(np.mean(f_det_mc), color='red', linestyle='--', alpha=0.7)
    ax.axvline(np.mean(f_final_mc), color='blue', linestyle='--', alpha=0.7)
    
    ax.set_xlabel('Objective Value')
    ax.set_ylabel('Density')
    ax.set_title('Det. vs Robust Distributions (MC)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('GMM-UQPCE Optimization Example', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('gmm_results.png', dpi=150)
    print("\nPlot saved to 'gmm_results.png'")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Design shift: Δx = {x_uq-x_det:.3f}, Δy = {y_uq-y_det:.3f}")
    print(f"Distance moved: {np.sqrt((x_uq-x_det)**2 + (y_uq-y_det)**2):.3f}")
    
    if abs(np.sqrt(f_var) - np.std(f_final_mc)) > 0.1:
        print(f"\n⚠️  WARNING: Significant discrepancy between PCE and MC std!")
        print(f"   This may indicate the PCE order ({order}) is too low for this problem.")
        print(f"   Consider increasing the order in input.yaml.")
    
    print("="*70 + "\n")


if __name__ == '__main__':
    main()