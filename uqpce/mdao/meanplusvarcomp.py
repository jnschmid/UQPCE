import numpy as np
import openmdao.api as om


class MeanPlusVarComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare(
            'variance_weight', types=(int, float), default=1, lower=0,
            upper=None, desc='Reference scale for 1 of the sample data'
        )

    def setup(self):

        self.add_input('mean', shape=(1,))
        self.add_input('variance', shape=(1,))
        self.add_output('mean_plus_var', shape=(1,))

        self.declare_partials(of='mean_plus_var', wrt='mean', val=1)
        self.declare_partials(of='mean_plus_var', wrt='variance', val=1)

        self._no_check_partials = True

    def compute(self, inputs, outputs):
        lmbd = self.options['variance_weight']
        outputs['mean_plus_var'] = inputs['mean'] + lmbd * inputs['variance']

if __name__ == '__main__':

    prob = om.Problem()
    prob.model.add_subsystem(
        'comp', MeanPlusVarComp(), promotes_inputs=['*'], promotes_outputs=['*']
    )

    prob.setup(force_alloc_complex=True)
    prob.set_val('mean', 4.7)
    prob.set_val('variance', 15)

    prob.run_model()
    prob.check_partials(compact_print=True, method='cs')
