import unittest
import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
from uqpce.mdao import interface
from pathlib import Path

# Import all components from main.py
from uqpce.examples.sellar.main import SellarDis1, SellarDis2, SellarObj, SellarConst1, SellarConst2
from uqpce.examples.sellar import main

class TestSellarComponents(unittest.TestCase):
    
    def setUp(self):
        """Set up test with actual UQPCE data"""
        main_dir = Path(main.__file__).parent
        input_file = str(main_dir / 'input.yaml')
        matrix_file = str(main_dir / 'run_matrix.dat')
        
        # Load UQPCE data
        _, _, _, _, _, resp_cnt, _, _, _, run_matrix = interface.initialize(input_file, matrix_file)
        
        self.vec_size = resp_cnt 
        self.run_matrix = run_matrix
        
        # Test with first sample - corrected values from run_matrix.dat
        self.a0_test = self.run_matrix[0, 0] 
        self.a1_test = self.run_matrix[0, 1]  
        self.a2_test = self.run_matrix[0, 2] 
        self.a3_test = self.run_matrix[0, 3] 
        
    def test_sellar_dis1(self):
        """Test SellarDis1 computation and derivatives"""
        prob = om.Problem(reports=None)
        prob.model.add_subsystem('dis1', SellarDis1(vec_size=1))
        prob.setup(force_alloc_complex=True)
        
        # Set inputs
        z1, z2 = 5.0, 2.0
        x = 1.0
        y2 = 12.0
        
        prob.set_val('dis1.z', [z1, z2])
        prob.set_val('dis1.x', x)
        prob.set_val('dis1.y2', [y2])
        prob.set_val('dis1.a0', [self.a0_test])
        prob.set_val('dis1.a1', [self.a1_test])
        prob.set_val('dis1.a2', [self.a2_test])
        
        prob.run_model()
        
        # Check computation: y1 = z1**2*a0 + z2*a1 + x*a2 - 0.2*y2
        expected_y1 = z1**2 * self.a0_test + z2 * self.a1_test + x * self.a2_test - 0.2 * y2
        assert_near_equal(prob.get_val('dis1.y1'), [expected_y1], tolerance=1e-10)
        
        # Check derivatives
        data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(data, atol=1e-10, rtol=1e-10)
        
    def test_sellar_dis2(self):
        """Test SellarDis2 computation and derivatives"""
        prob = om.Problem(reports=None)
        prob.model.add_subsystem('dis2', SellarDis2(vec_size=1))
        prob.setup(force_alloc_complex=True)
        
        # Set inputs
        z1, z2 = 5.0, 2.0
        y1 = 25.6
        
        prob.set_val('dis2.z', [z1, z2])
        prob.set_val('dis2.y1', [y1])
        prob.set_val('dis2.a0', [self.a0_test])
        prob.set_val('dis2.a1', [self.a1_test])
        
        prob.run_model()
        
        # Check computation: y2 = y1**0.5 + z1*a0 + z2*a1
        expected_y2 = y1**0.5 + z1 * self.a0_test + z2 * self.a1_test
        assert_near_equal(prob.get_val('dis2.y2'), [expected_y2], tolerance=1e-10)
        
        # Check derivatives
        data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(data, atol=1e-10, rtol=1e-10)
        
    def test_sellar_obj(self):
        """Test SellarObj computation and derivatives"""
        prob = om.Problem(reports=None)
        prob.model.add_subsystem('obj', SellarObj(vec_size=1))
        prob.setup(force_alloc_complex=True)
        
        # Set inputs
        x = 1.0
        z1, z2 = 5.0, 2.0
        y1 = 25.6
        y2 = 12.0
        
        prob.set_val('obj.x', x)
        prob.set_val('obj.z', [z1, z2])
        prob.set_val('obj.y1', [y1])
        prob.set_val('obj.y2', [y2])
        prob.set_val('obj.a0', [self.a0_test])
        prob.set_val('obj.a1', [self.a1_test])
        prob.set_val('obj.a2', [self.a2_test])
        prob.set_val('obj.a3', [self.a3_test])
        
        prob.run_model()
        
        # Check computation: obj = x**2*a2 + z2*a1 + y1*a3 + exp(-y2)*a0
        expected_obj = x**2 * self.a2_test + z2 * self.a1_test + y1 * self.a3_test + np.exp(-y2) * self.a0_test
        assert_near_equal(prob.get_val('obj.obj'), [expected_obj], tolerance=1e-10)
        
        # Check derivatives
        data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(data, atol=1e-10, rtol=1e-10)
        
    def test_sellar_const1(self):
        """Test SellarConst1 computation and derivatives"""
        prob = om.Problem(reports=None)
        prob.model.add_subsystem('const1', SellarConst1(vec_size=1))
        prob.setup(force_alloc_complex=True)
        
        # Set input
        y1 = 25.6
        prob.set_val('const1.y1', [y1])
        
        prob.run_model()
        
        # Check computation: const1 = 8 - y1
        expected_const1 = 8 - y1
        assert_near_equal(prob.get_val('const1.const1'), [expected_const1], tolerance=1e-10)
        
        # Check derivatives
        data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(data, atol=1e-10, rtol=1e-10)
        
    def test_sellar_const2(self):
        """Test SellarConst2 computation and derivatives"""
        prob = om.Problem(reports=None)
        prob.model.add_subsystem('const2', SellarConst2(vec_size=1))
        prob.setup(force_alloc_complex=True)
        
        # Set input
        y2 = 12.0
        prob.set_val('const2.y2', [y2])
        
        prob.run_model()
        
        # Check computation: const2 = y2 - 12
        expected_const2 = y2 - 12
        assert_near_equal(prob.get_val('const2.const2'), [expected_const2], tolerance=1e-10)
        
        # Check derivatives
        data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(data, atol=1e-10, rtol=1e-10)
        
    def test_vectorized_components(self):
        """Test all components with full vector size"""
        # Test SellarDis1 with vectors
        prob1 = om.Problem(reports=None)
        prob1.model.add_subsystem('dis1', SellarDis1(vec_size=self.vec_size))
        prob1.setup()
        
        prob1.set_val('dis1.z', [5.0, 2.0])
        prob1.set_val('dis1.x', 1.0)
        prob1.set_val('dis1.y2', np.ones(self.vec_size) * 12.0)
        prob1.set_val('dis1.a0', self.run_matrix[:, 0])
        prob1.set_val('dis1.a1', self.run_matrix[:, 1])
        prob1.set_val('dis1.a2', self.run_matrix[:, 2])
        
        prob1.run_model()
        
        y1 = prob1.get_val('dis1.y1')
        self.assertEqual(y1.shape, (self.vec_size,))
        self.assertTrue(np.all(np.isfinite(y1)))
        
        # Test SellarObj with vectors
        prob2 = om.Problem(reports=None)
        prob2.model.add_subsystem('obj', SellarObj(vec_size=self.vec_size))
        prob2.setup()
        
        prob2.set_val('obj.x', 1.0)
        prob2.set_val('obj.z', [5.0, 2.0])
        prob2.set_val('obj.y1', np.ones(self.vec_size) * 25.6)
        prob2.set_val('obj.y2', np.ones(self.vec_size) * 12.0)
        prob2.set_val('obj.a0', self.run_matrix[:, 0])
        prob2.set_val('obj.a1', self.run_matrix[:, 1])
        prob2.set_val('obj.a2', self.run_matrix[:, 2])
        prob2.set_val('obj.a3', self.run_matrix[:, 3])
        
        prob2.run_model()
        
        obj = prob2.get_val('obj.obj')
        self.assertEqual(obj.shape, (self.vec_size,))
        self.assertTrue(np.all(np.isfinite(obj)))
        
    def test_edge_cases(self):
        """Test components with edge case values"""
        # Test with y1 = 0 for SellarDis2 (sqrt(0) = 0)
        prob = om.Problem(reports=None)
        prob.model.add_subsystem('dis2', SellarDis2(vec_size=1))
        prob.setup()
        
        prob.set_val('dis2.z', [1.0, 1.0])
        prob.set_val('dis2.y1', [0.0])
        prob.set_val('dis2.a0', [1.0])
        prob.set_val('dis2.a1', [1.0])
        
        prob.run_model()
        
        # y2 = 0**0.5 + 1*1 + 1*1 = 0 + 1 + 1 = 2
        expected_y2 = 2.0
        assert_near_equal(prob.get_val('dis2.y2'), [expected_y2], tolerance=1e-10)
        
        # Test with large negative y2 for SellarObj (exp(-y2) should be very large)
        prob2 = om.Problem(reports=None)
        prob2.model.add_subsystem('obj', SellarObj(vec_size=1))
        prob2.setup()
        
        prob2.set_val('obj.x', 0.0)
        prob2.set_val('obj.z', [0.0, 0.0])
        prob2.set_val('obj.y1', [0.0])
        prob2.set_val('obj.y2', [-10.0]) 
        prob2.set_val('obj.a0', [1.0])
        prob2.set_val('obj.a1', [1.0])
        prob2.set_val('obj.a2', [1.0])
        prob2.set_val('obj.a3', [1.0])
        
        prob2.run_model()
        
        # obj = 0*1 + 0*1 + 0*1 + exp(10)*1 ≈ 22026
        obj_val = prob2.get_val('obj.obj')[0]
        self.assertAlmostEqual(obj_val, np.exp(10), places=5)
        
    def test_actual_sample_values(self):
        """Test with multiple actual samples from run_matrix.dat"""
        # Test first 5 samples
        for i in range(min(5, self.vec_size)):
            with self.subTest(sample=i):
                prob = om.Problem(reports=None)
                model = prob.model
                
                # Add all components
                model.add_subsystem('dis1', SellarDis1(vec_size=1))
                model.add_subsystem('dis2', SellarDis2(vec_size=1))
                model.add_subsystem('obj', SellarObj(vec_size=1))
                
                prob.setup()
                
                # Set design variables
                z1, z2 = 5.0, 2.0
                x = 1.0
                
                # Get sample values
                a0 = self.run_matrix[i, 0]
                a1 = self.run_matrix[i, 1]
                a2 = self.run_matrix[i, 2]
                a3 = self.run_matrix[i, 3]
                
                # Initial y2 guess
                y2_init = 12.0
                
                # Set inputs for dis1
                prob.set_val('dis1.z', [z1, z2])
                prob.set_val('dis1.x', x)
                prob.set_val('dis1.y2', [y2_init])
                prob.set_val('dis1.a0', [a0])
                prob.set_val('dis1.a1', [a1])
                prob.set_val('dis1.a2', [a2])
                
                prob.run_model()
                y1 = prob.get_val('dis1.y1')[0]
                
                # Verify y1 calculation
                expected_y1 = z1**2 * a0 + z2 * a1 + x * a2 - 0.2 * y2_init
                self.assertAlmostEqual(y1, expected_y1, places=10)
                
                # Set inputs for dis2
                prob.set_val('dis2.z', [z1, z2])
                prob.set_val('dis2.y1', [y1])
                prob.set_val('dis2.a0', [a0])
                prob.set_val('dis2.a1', [a1])
                
                prob.run_model()
                y2 = prob.get_val('dis2.y2')[0]
                
                # Verify y2 calculation
                expected_y2 = y1**0.5 + z1 * a0 + z2 * a1
                self.assertAlmostEqual(y2, expected_y2, places=10)
                
                # Test objective
                prob.set_val('obj.x', x)
                prob.set_val('obj.z', [z1, z2])
                prob.set_val('obj.y1', [y1])
                prob.set_val('obj.y2', [y2])
                prob.set_val('obj.a0', [a0])
                prob.set_val('obj.a1', [a1])
                prob.set_val('obj.a2', [a2])
                prob.set_val('obj.a3', [a3])
                
                prob.run_model()
                obj = prob.get_val('obj.obj')[0]
                
                # Verify objective calculation
                expected_obj = x**2 * a2 + z2 * a1 + y1 * a3 + np.exp(-y2) * a0
                self.assertAlmostEqual(obj, expected_obj, places=10)


if __name__ == '__main__':
    unittest.main(verbosity=2)