import numpy as np
import openmdao.api as om


class Obj(om.ExplicitComponent):
    """
    OpenMDAO Explicit Component which defines the problem objective.
    The goal is for the trajectory mean to be a minimum distance from
    the target value.
    """
    def initialize(self):
        self.options.declare('num_samples', types=int)

    def setup(self):
        self.add_input('mean')
        self.add_input('variance')

        self.add_output('obj')

    def setup_partials(self):
        self.declare_partials(of='obj', wrt='mean')

    def compute(self, inputs, outputs):
        tgt = 150

        outputs['obj'] = (tgt - inputs['mean']) ** 2

    def compute_partials(self, inputs, partials):
        tgt = 150

        partials['obj', 'mean'] = -2 * (tgt - inputs['mean'])
