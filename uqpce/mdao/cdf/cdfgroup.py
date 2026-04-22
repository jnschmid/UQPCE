import numpy as np
import openmdao.api as om
import jax
import jax.numpy as jnp

from uqpce.mdao.cdf.cdfresidcomp import CDFResidComp


class CDFComp(om.JaxExplicitComponent):
    """
    Component class to calculate the differentiable confidence interval.
    """

    def initialize(self):
        self.options.declare('vec_size', types=int)

        # The probability of the response is greater than the 1-alpha value
        # i.e. alpha=0.05 corresponds to the cumulative probability of 95%
        self.options.declare(
            'alpha', types=float, default=0.05,
            desc='Single-sided upper confidence interval of (1-alpha)'
        )
        self.options.declare('aleatory_cnt', types=int, allow_none=False)
        self.options.declare('epistemic_cnt', types=int, allow_none=False)
        self.options.declare(
            'tail', values=['lower', 'upper'], allow_none=False
        )

        self._no_check_partials = True

    def setup(self):
        alpha = self.options['alpha']
        aleat_cnt = self.options['aleatory_cnt']
        epist_cnt = self.options['epistemic_cnt']

        self.add_input('samples', shape=(epist_cnt*aleat_cnt,))
        self.add_output('f_ci', shape=(epist_cnt,))

        self._sig = (1-alpha/2) if self.options['tail'] == 'upper' else alpha/2

    def get_self_statics(self):
        return (
            self.options['alpha'], self.options['epistemic_cnt'],
            self.options['aleatory_cnt']
        )

    def compute_primal(self, samples):
        aleat_cnt = self.options['aleatory_cnt']
        epist_cnt = self.options['epistemic_cnt']
        tail = self.options['tail']

        samps = jnp.reshape(samples, (-1, aleat_cnt))

        if aleat_cnt != 1:
            f_ci = jnp.quantile(jnp.real(samps), self._sig, axis=1)

            # Mixed uncertainty
            if epist_cnt != 1:
                if tail == 'upper':
                    f_ci = jnp.max(f_ci)
                else:
                    f_ci = jnp.min(f_ci)
        else:  # Pure epistemic
            if tail == 'upper':
                f_ci = jnp.max(samps)
            else:
                f_ci = jnp.min(samps)

        return f_ci


class CDFGroup(om.Group):

    def initialize(self):
        self.options.declare('vec_size', types=int)
        self.options.declare(
            'alpha', types=float, default=0.05,
            desc='Single-sided upper confidence interval of (1-alpha)'
        )
        self.options.declare('tanh_omega', types=float, default=1e-6)
        self.options.declare(
            'tail', values=['lower', 'upper'], allow_none=False
        )
        self.options.declare('aleatory_cnt', types=int, allow_none=False)
        self.options.declare('epistemic_cnt', types=int, allow_none=False)
        self.options.declare(
            'sample_ref0', types=(float, int), default=0.0,
            desc='Scaling parameter. The value in the user-defined units of '
            'this output variable when the scaled value is 0. Default is 0.'
        )
        self.options.declare(
            'sample_ref', types=(float, int), default=1.0,
            desc='Scaling parameter. The value in the user-defined units of '
            'this output variable when the scaled value is 1. Default is 1.'
        )

    def setup(self):

        vec_size = self.options['vec_size']
        alpha = self.options['alpha']
        tanh_omega = self.options['tanh_omega']
        tail = self.options['tail']
        aleat_cnt = self.options['aleatory_cnt']
        epist_cnt = self.options['epistemic_cnt']
        sample_ref0 = self.options['sample_ref0']
        sample_ref = self.options['sample_ref']

        self.add_subsystem(
            'cdf', CDFResidComp(
                vec_size=vec_size, alpha=alpha, tanh_omega=tanh_omega,
                tail=tail, aleatory_cnt=aleat_cnt, epistemic_cnt=epist_cnt,
                sample_ref0=sample_ref0, sample_ref=sample_ref
            ),
            promotes_inputs=[('samples', 'f_sampled'), 'f_ci'],
            promotes_outputs=['ci_resid']
        )

        bal = self.add_subsystem(
            'bal', om.BalanceComp(val=jnp.ones([epist_cnt])),
            promotes_inputs=['ci_resid'], promotes_outputs=['f_ci']
        )
        bal.add_balance(
            name='f_ci', lhs_name='ci_resid', val=jnp.ones([epist_cnt])
        )

        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        self.linear_solver = om.DirectSolver()

        minimum = (self.options['tail'] == 'lower')

        if vec_size == aleat_cnt:  # purely aleatoric
            pass
        else:
            self.add_subsystem(
                'ks', om.KSComp(width=epist_cnt, minimum=minimum, rho=1000.),
                promotes_outputs=[('KS', 'ci')]
            )
            self.connect('f_ci', 'ks.g')

    def guess_nonlinear(self, inputs, outputs, residuals):
        aleatory_cnt = self.options['aleatory_cnt']
        samples = inputs['cdf.samples']
        x = jnp.reshape(samples, (-1, aleatory_cnt))
        outputs['f_ci'] = jnp.percentile(  # find CI of curves
            x, self._get_subsystem('cdf')._sig*100, axis=1
        )
        self._truth = jnp.max(outputs['f_ci'])


if __name__ == '__main__':

    from scipy.stats import binom, norm

    lower = -2
    upper = 2

    alpha = 0.05
    aleat_cnt = 40
    epist_cnt = 1
    vec_size = aleat_cnt * epist_cnt

    np.random.seed(1)
    samps = (
        binom.rvs(n=5, p=0.3, size=vec_size)
        + norm.rvs(0, 0.05, size=vec_size)
    )

    prob = om.Problem()
    prob.model.add_subsystem(
        'comp',
        CDFComp(
            alpha=alpha, tail='upper', vec_size=vec_size,
            epistemic_cnt=epist_cnt, aleatory_cnt=aleat_cnt
        ),
        promotes_inputs=['*'], promotes_outputs=['*']
    )

    prob.setup(force_alloc_complex=True)
    prob.set_val('samples', samps)
    prob.run_model()

    with np.printoptions(threshold=np.inf):
        partials = prob.check_partials(compact_print=False, method='fd', form='central')

    print(np.quantile(samps, [0.025, 1-0.025]))
    print(prob.get_val('f_ci'))
