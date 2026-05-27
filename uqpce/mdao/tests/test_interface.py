import unittest

from uqpce.mdao.interface import initialize, initialize_dict


class TestInterface(unittest.TestCase):
    def setUp(self):
        pass

    def test_initialize(self):
        input_file = 'uqpce/examples/GMM/input.yaml'
        matrix_file = 'uqpce/examples/GMM/run_matrix.dat'
        init_args = initialize(input_file=input_file, matrix_file=matrix_file, order=1)

        self.assertTrue(
            init_args[0].shape == (50, 4),
            msg='Function `initialize` is not correct. It should be the `var_basis_sys_eval`.'
        )
        self.assertTrue(
            init_args[1].shape == (4, 1),
            msg='Function `initialize` is not correct. It should be the `norm_sq`.'
        )
        self.assertTrue(
            init_args[2].shape == (1000000, 4),
            msg='Function `initialize` is not correct. It should be the `var_basis_resamp`.'
        )
        self.assertTrue(
            init_args[3] == 1000000,
            msg='Function `initialize` is not correct. It should be the `aleat_samps`.'
        )
        self.assertTrue(
            init_args[4] == 1,
            msg='Function `initialize` is not correct. It should be the `epist_samps`.'
        )
        self.assertTrue(
            init_args[5] == 50,
            msg='Function `initialize` is not correct. It should be the `resp_count`.'
        )
        self.assertTrue(
            init_args[6] == 1,
            msg='Function `initialize` is not correct. It should be the `order`.'
        )
        self.assertTrue(
            init_args[7].shape == (3,),
            msg='Function `initialize` is not correct. It should be the `variables`.'
        )
        self.assertTrue(
            init_args[8] == 0.05,
            msg='Function `initialize` is not correct. It should be the `significance`.'
        )
        self.assertTrue(
            init_args[9].shape == (50, 3),
            msg='Function `initialize` is not correct. It should be the `samples`.'
        )

    def test_initialize_dict(self):
        input_file = 'uqpce/examples/GMM/input.yaml'
        matrix_file = 'uqpce/examples/GMM/run_matrix.dat'
        init_args = initialize_dict(input_file=input_file, matrix_file=matrix_file, order=1)

        self.assertTrue(
            init_args['var_basis'].shape == (50, 4),
            msg='Function `initialize` is not correct. It should be the `var_basis`.'
        )
        self.assertTrue(
            init_args['norm_sq'].shape == (4, 1),
            msg='Function `initialize` is not correct. It should be the `norm_sq`.'
        )
        self.assertTrue(
            init_args['resampled_var_basis'].shape == (1000000, 4),
            msg='Function `initialize` is not correct. It should be the `resampled_var_basis`.'
        )
        self.assertTrue(
            init_args['aleatory_cnt'] == 1000000,
            msg='Function `initialize` is not correct. It should be the `aleatory_cnt`.'
        )
        self.assertTrue(
            init_args['epistemic_cnt'] == 1,
            msg='Function `initialize` is not correct. It should be the `epistemic_cnt`.'
        )
        self.assertTrue(
            init_args['resp_cnt'] == 50,
            msg='Function `initialize` is not correct. It should be the `resp_cnt`.'
        )
        self.assertTrue(
            init_args['order'] == 1,
            msg='Function `initialize` is not correct. It should be the `order`.'
        )
        self.assertTrue(
            init_args['variables'].shape == (3,),
            msg='Function `initialize` is not correct. It should be the `variables`.'
        )
        self.assertTrue(
            init_args['significance'] == 0.05,
            msg='Function `initialize` is not correct. It should be the `significance`.'
        )
        self.assertTrue(
            init_args['run_matrix'].shape == (50, 3),
            msg='Function `initialize` is not correct. It should be the `run_matrix`.'
        )
        self.assertTrue(
            init_args['model_matrix'].shape == (4, 3),
            msg='Function `initialize` is not correct. It should be the `model_matrix`.'
        )


if __name__ == '__main__':
    suite = unittest.TestSuite()
    unittest.main()