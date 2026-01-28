
import unittest
import torch
import config
import numpy as np

class TestConfig(unittest.TestCase):
    def test_torch_geometry_conversion(self):
        """
        Test that config.py can provide geometry as Torch tensors.
        This tests the new functionality to be added.
        """
        # Check if the function exists
        if not hasattr(config, 'get_torch_geometry'):
            self.fail("config.get_torch_geometry not implemented yet")
        
        # Call it
        device = torch.device('cpu')
        geo_tensors = config.get_torch_geometry(device=device)
        
        # Verify structure
        self.assertIn('TX1', geo_tensors)
        self.assertIn('RX1', geo_tensors)
        self.assertIsInstance(geo_tensors['TX1']['position'], torch.Tensor)
        self.assertEqual(geo_tensors['TX1']['position'].dtype, torch.float32)
        
        # Verify values match the numpy source
        tx1_np = config.RADAR_GEOMETRY['TX1']['position']
        tx1_torch = geo_tensors['TX1']['position']
        
        np.testing.assert_allclose(tx1_np, tx1_torch.numpy(), rtol=1e-5)

if __name__ == '__main__':
    unittest.main()
