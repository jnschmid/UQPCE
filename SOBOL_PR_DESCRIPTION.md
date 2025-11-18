# Add Sobol Sensitivity Analysis to UQPCE

## Description
This PR exposes UQPCE's existing Sobol sensitivity analysis capabilities through the OpenMDAO interface, making Sobol indices natively accessible as OpenMDAO outputs for use in optimization and analysis workflows.

## Motivation
UQPCE already computes Sobol sensitivity indices internally, but these were not accessible through the OpenMDAO interface. This PR bridges that gap by creating an OpenMDAO component that exposes these indices as standard outputs, enabling their use in gradient-based optimization and sensitivity studies.

## Implementation

### New Component: SobolComp
- **Location**: `uqpce/mdao/sobolcomp.py`
- **Functionality**:
  - Wraps existing `uqpce.pce._helpers.calc_sobols()` and `create_total_sobols()` functions
  - Exposes individual and total Sobol indices as OpenMDAO outputs
  - Provides analytic derivatives for gradient-based optimization
  - Integrates seamlessly with UQPCEGroup

### Modified Files
1. **`uqpce/mdao/interface.py`**
   - Returns `model_matrix` from `initialize()` function
   - The model matrix is needed by SobolComp to compute total Sobols

2. **`uqpce/mdao/uqpcegroup.py`**
   - Added `compute_sobols` option (default: False)
   - Added `model_matrix` option for Sobol computation
   - Automatically adds SobolComp when `compute_sobols=True`

## Technical Details

### Mathematical Formulation
For a PCE expansion: `f = Σ c_i Ψ_i(ξ)`

Individual Sobol indices:
```
S_i = (c_i² × ||Ψ_i||²) / Var(f)
```

Total Sobol indices:
```
S_T,j = Σ S_i for all terms i where variable j appears
```

### Key Features
- **Native OpenMDAO outputs**: Sobol indices available as `response:sobols` and `response:total_sobols`
- **Analytic derivatives**: Exact gradients for optimization workflows
- **Optional activation**: Only computes when `compute_sobols=True`

## Usage Example

```python
from uqpce.mdao import interface
from uqpce.mdao.uqpcegroup import UQPCEGroup

# Initialize UQPCE (now returns model_matrix)
(var_basis, norm_sq, resampled_var_basis,
 aleatory_cnt, epistemic_cnt, resp_cnt, order, variables,
 sig, run_matrix, model_matrix) = interface.initialize(input_file, matrix_file)

# Add UQPCEGroup with Sobol computation
model.add_subsystem(
    'uqpce',
    UQPCEGroup(
        var_basis=var_basis,
        norm_sq=norm_sq,
        resampled_var_basis=resampled_var_basis,
        model_matrix=model_matrix,
        compute_sobols=True,  # Enable Sobol indices
        # ... other options
    )
)

# Access Sobol indices after running
sobols = prob.get_val('response:sobols')  # Individual indices
total_sobols = prob.get_val('response:total_sobols')  # Total indices per variable
```

## Testing
- Created test suite in `test_sobol_integration.py`
- Tests verify:
  - Correct computation of Sobol indices
  - Analytic derivatives match finite differences
  - Integration with UQPCEGroup

## Example Output
```
Individual Sobols: [0.658, 0.237, 0.105]  # Per PCE term
Total Sobols: [0.763, 0.342]  # Per uncertain variable
```