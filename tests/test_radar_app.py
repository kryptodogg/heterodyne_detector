
import unittest
import torch
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

if __name__ == '__main__':
    unittest.main()
