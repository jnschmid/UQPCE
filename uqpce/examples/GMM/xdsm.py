from pyxdsm.XDSM import XDSM, OPT, FUNC

def generate_xdsm():
    xdsm = XDSM()

    # Add systems
    xdsm.add_system('opt', OPT, r'\textbf{Optimizer}')
    xdsm.add_system('quadratic', FUNC, [r'\textbf{Quadratic Function}',
                                        r'f = a_1 x^2 + a_2 x + a_3'])
    xdsm.add_system('constraint', FUNC, [r'\textbf{Constraint}',
                                         r'g = 5 - f'])
    xdsm.add_system('UQPCE', FUNC, [r'\textbf{UQPCE}',
                                    r'Uncertainty Quantification'])

    # Design variable
    xdsm.add_input('opt', r'x')
    
    # Uncertain parameters
    xdsm.add_input('quadratic', r'a_1^\dagger (GMM), a_2^\dagger (Uniform), a_3^\dagger (Lognormal)')

    # Connections
    xdsm.connect('opt', 'quadratic', r'x')
    xdsm.connect('quadratic', 'constraint', r'f^\dagger')
    xdsm.connect('quadratic', 'UQPCE', r'f^\dagger')
    xdsm.connect('constraint', 'UQPCE', r'g^\dagger')
    xdsm.connect('UQPCE', 'opt', r'f_{\mu+\sigma^2}, g_{CI_{lower}}')

    # Outputs
    xdsm.add_output('opt', r'x^*', side='right')
    xdsm.add_output('UQPCE', r'f_{CI}, g_{CI}', side='right')

    # Write the XDSM
    xdsm.write("gmm_example_xdsm")

if __name__ == "__main__":
    generate_xdsm()