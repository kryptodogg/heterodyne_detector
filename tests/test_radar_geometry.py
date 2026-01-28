
import unittest
import torch
import numpy as np
import config

# We expect RadarGeometry to be available in main.py (as per spec)
# OR we can put it in a new module. The spec says "Implement RadarGeometry in main.py or a local module".
# Given the "Modular Constraint" of 250-500 lines, and main.py being an orchestrator, 
# it's better to put it in a separate module if main.py is getting large.
# However, the user explicitly asked: "Refactor Core Pipeline... Implement the RadarGeometry class directly in main.py (or a dedicated internal module)"
# and the original plan was "Implement RadarGeometry in main.py or a local module".
# Let's check main.py size.

class TestRadarGeometry(unittest.TestCase):
    def test_initialization(self):
        """Test RadarGeometry initialization from config"""
        # We need to import RadarGeometry. Let's assume it will be in 'radar_geometry.py' for better modularity
        # or we can put it in 'main.py' if requested.
        # The prompt said "How should TX/RX geometry be stored in main.py?" and I suggested "Insert this class near the top of main.py".
        # So I will test importing from main.
        
        try:
            from main import RadarGeometry
        except ImportError:
            self.fail("Could not import RadarGeometry from main")

        device = torch.device('cpu')
        geo = RadarGeometry(config.RADAR_GEOMETRY, device=device)
        
        self.assertIsInstance(geo.tx1_pos, torch.Tensor)
        self.assertIsInstance(geo.rx1_pos, torch.Tensor)
        self.assertTrue(hasattr(geo, 'baseline'))
        self.assertTrue(hasattr(geo, 'wavelength'))

    def test_steering_vector(self):
        """Test steering vector calculation"""
        try:
            from main import RadarGeometry
        except ImportError:
            self.fail("Could not import RadarGeometry from main")
            
        device = torch.device('cpu')
        geo = RadarGeometry(config.RADAR_GEOMETRY, device=device)
        
        # Test 0 degrees (broadside)
        theta = torch.tensor([0.0])
        sv = geo.compute_steering_vector(theta)
        
        # For 0 degrees, path difference is 0 (relative to reference if broadside)
        # But wait, compute_steering_vector depends on the array manifold logic.
        # Let's assume a standard implementation we will write.
        
        self.assertIsInstance(sv, torch.Tensor)
        self.assertEqual(sv.shape[1], 4) # 2TX * 2RX = 4 virtual elements
        # The original code used 2TX2RX. 
        # For now, let's just ensure it runs and returns complex tensors.
        self.assertTrue(torch.is_complex(sv))

if __name__ == '__main__':
    unittest.main()
