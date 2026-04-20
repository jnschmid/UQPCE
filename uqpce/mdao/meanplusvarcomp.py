import openmdao.api as om
import numpy as np

class MeanPlusVarComp(om.ExplicitComponent):

    def setup(self):

        self.add_input('mean', shape=(1,), units_by_conn=True, units=None)
        self.add_input('variance', shape=(1,), units_by_conn=True, units=None)
        self.add_output('mean_plus_var', shape=(1,), copy_units='mean')

        self.declare_partials(of='mean_plus_var', wrt='mean', val=1)
        self.declare_partials(of='mean_plus_var', wrt='variance', val=1)

        self._no_check_partials = True

    def compute(self, inputs, outputs):

        outputs['mean_plus_var'] = inputs['mean'] + inputs['variance']


if __name__ == '__main__':

    prob = om.Problem()
    prob.model.add_subsystem(
        'add_variance_units',
        om.ExecComp('variance = 1*var', variance={'units': None}, shape=(1,)),
        promotes_inputs=['var'], promotes_outputs=['variance']
    )
    prob.model.add_subsystem(
        'add_mean_units',
        om.ExecComp('mean = 1*mu', mean={'units': None}, shape=(1,)),
        promotes_inputs=['mu'], promotes_outputs=['mean']
    )
    prob.model.add_subsystem(
        'comp', MeanPlusVarComp(), promotes_inputs=['mean', 'variance'], promotes_outputs=['*']
    )
    # knowns = {n for n, d in graph.nodes(data=True) if (d[prop] is not None) or d[prop_by_conn] or d[copy_prop]} # JOANNA

    prob.setup(force_alloc_complex=True)
    prob.set_val('mean', 4.7)
    prob.set_val('variance', 15)

    prob.run_model()
    prob.check_partials(compact_print=True, method='cs')
