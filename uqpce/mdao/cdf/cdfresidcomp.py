import numpy as np
import openmdao.api as om
from openmdao.jax import act_tanh
import jax.numpy as jnp


class CDFResidComp(om.JaxExplicitComponent):
    """
    Component class to calculate the residual between the current tanh function
    evaluation sum and the confidence-interval-based value that it is desired
    to be.
    """

    def initialize(self):
        self.options.declare('vec_size', types=int)

        # The probability of the response is greater than the 1-alpha value
        # i.e. alpha=0.05 corresponds to the cumulative probability of 95%
        self.options.declare(
            'alpha', types=float, default=0.05,
            desc='Single-sided upper confidence interval of (1-alpha)'
        )
        self.options.declare('tanh_omega', types=float, default=1e-6)
        self.options.declare('aleatory_cnt', types=int, allow_none=False)
        self.options.declare('epistemic_cnt', types=int, allow_none=False)
        self.options.declare(
            'tail', values=['lower', 'upper'], allow_none=False
        )
        self.options.declare(
            'sample_ref0', types=(float, int), default=0.0,
            desc='Reference scale for 0 of the sample data'
        )
        self.options.declare(
            'sample_ref', types=(float, int), default=1.0,
            desc='Reference scale for 1 of the sample data'
        )

        self._no_check_partials = True

    def setup(self):
        alpha = self.options['alpha']
        aleat_cnt = self.options['aleatory_cnt']
        epist_cnt = self.options['epistemic_cnt']

        self.add_input('samples', shape=(epist_cnt*aleat_cnt,), units_by_conn=True)
        self.add_input('f_ci', shape=(epist_cnt,), copy_units='samples')

        self.add_output('ci_resid', shape=(epist_cnt,), copy_units='samples')

        self._sig = (1-alpha/2) if self.options['tail'] == 'upper' else alpha/2

    def get_self_statics(self):
        return (
            self.options['alpha'], self.options['tanh_omega'],
            self.options['aleatory_cnt'], self.options['sample_ref0'],
            self.options['sample_ref'],
        )

    def compute_primal(self, samples, f_ci):
        sample_ref0 = self.options['sample_ref0']
        sample_ref = self.options['sample_ref']
        aleat_cnt = self.options['aleatory_cnt']
        tanh_omega = self.options['tanh_omega']

        f_sampled = (samples - sample_ref0) / sample_ref
        f_ci = (f_ci - sample_ref0) / sample_ref

        x = jnp.transpose(jnp.reshape(f_sampled, (-1, aleat_cnt)))
        dlt = jnp.transpose(act_tanh(x, mu=tanh_omega, z=f_ci, a=1, b=0))

        return (jnp.sum(dlt, axis=1) / aleat_cnt) - self._sig


if __name__ == '__main__':

    lower = -2
    upper = 2

    alpha = 0.05
    epist_cnt = 1
    aleat_cnt = 10000
    vec_size = aleat_cnt*epist_cnt

    np.random.seed(1)
    samps = np.random.uniform(low=lower, high=upper, size=vec_size)
    ci_init = np.quantile(samps.reshape(-1, aleat_cnt), 1-alpha/2)
    prob = om.Problem()
    prob.model.add_subsystem(
        'res_cdf', CDFResidComp(
            vec_size=vec_size, alpha=alpha, tanh_omega=0.001, tail='upper',
            aleatory_cnt=aleat_cnt, epistemic_cnt=epist_cnt
        ),
        promotes_inputs=['samples', 'f_ci'],
        promotes_outputs=['ci_resid']
    )

    bal = prob.model.add_subsystem(
        'bal', om.BalanceComp(val=np.ones([epist_cnt])),
        promotes_inputs=['ci_resid'], promotes_outputs=['f_ci']
    )
    bal.add_balance(name='f_ci', lhs_name='ci_resid', val=np.ones([epist_cnt]))

    prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
    prob.model.linear_solver = om.DirectSolver()

    prob.setup(force_alloc_complex=True)
    prob.set_val('res_cdf.samples', samps)
    prob.set_val('bal.f_ci', ci_init)
    prob.run_model()

    prob.check_partials(compact_print=True, method='cs')
