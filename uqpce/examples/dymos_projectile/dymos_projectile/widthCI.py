import openmdao.api as om


class WidthCI(om.ExplicitComponent):
    """
    OpenMDAO Explicit Component which computes the difference between
    upper and lower confidence interval bounds (with respect to distance traveled).
    Component behaves as a constraint.
    """
    def setup(self):
        self.add_input('ci_lower')
        self.add_input('ci_upper')
        self.add_output('width')

    def setup_partials(self):
        self.declare_partials(of='width', wrt='ci_lower')
        self.declare_partials(of='width', wrt='ci_upper')

    def compute(self, inputs, outputs):
        outputs['width'] = inputs['ci_upper'] - inputs['ci_lower']

    def compute_partials(self, inputs, partials):
        partials['width', 'ci_lower'] = -1.0
        partials['width', 'ci_upper'] = 1.0
