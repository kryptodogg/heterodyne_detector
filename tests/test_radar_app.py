
import unittest
import torch
import numpy as np
from main import RadarApp
from config import GPU_CONFIG

class TestRadarApp(unittest.TestCase):
    def test_init_buffers(self):
        """
        Test that RadarApp initializes buffers on the correct device.
        """
        # Initialize app in simulation mode to avoid hardware requirements
        app = RadarApp(simulate=True, enable_viz=False)
        
        # Check if device is set
        self.assertIsInstance(app.device, torch.device)
        
        # Check if buffers are pre-allocated (this is the new feature)
        if not hasattr(app, 'rx1_buffer'):
            self.fail("RadarApp.rx1_buffer not pre-allocated")
            
        self.assertIsInstance(app.rx1_buffer, torch.Tensor)
        self.assertEqual(app.rx1_buffer.device.type, app.device.type)
        self.assertEqual(app.rx1_buffer.shape[0], GPU_CONFIG['buffer_size'])

    def test_process_buffer_torch(self):
        """
        Test that process_buffer handles Torch tensors correctly.
        """
        app = RadarApp(simulate=True, enable_viz=False)
        
        # Create synthetic data (NumPy)
        buffer_size = GPU_CONFIG['buffer_size']
        rx1_np = np.random.randn(buffer_size) + 1j * np.random.randn(buffer_size)
        rx2_np = np.random.randn(buffer_size) + 1j * np.random.randn(buffer_size)
        
        # Process
        results = app.process_buffer(rx1_np, rx2_np)
        
        # Verify results structure
        self.assertIn('detection', results)
        self.assertIn('mfcc_features', results)
        self.assertIn('pattern_matches', results)
        
        # Check if internal processing used Torch (this will be verified by sub-module behavior)
        # For now, let's just ensure it doesn't crash and returns expected types
        self.assertIsInstance(results['mfcc_features'], torch.Tensor)

if __name__ == '__main__':
    import numpy as np # Ensure numpy is available for the test
    unittest.main()
