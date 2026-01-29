
import unittest
import torch
import sys
from unittest.mock import MagicMock, patch

# Mock other dependencies that might cause issues during import or init
sys.modules['adi'] = MagicMock()
sys.modules['dash'] = MagicMock()
sys.modules['dash.dependencies'] = MagicMock()
sys.modules['dash.html'] = MagicMock()
sys.modules['dash.dcc'] = MagicMock()
sys.modules['plotly'] = MagicMock()
sys.modules['plotly.graph_objects'] = MagicMock()

# Import modules to test
from heterodyne_detector import HeterodyneDetector
from noise_canceller import SpatialNoiseCanceller
from config import RADAR_GEOMETRY, NOISE_CANCELLATION
from main import RadarGeometry

class TestGPUSupport(unittest.TestCase):

    def test_heterodyne_detector_device(self):
        """Verify HeterodyneDetector respects the device parameter"""
        device = torch.device('cpu') # Use CPU for actual test execution to avoid CUDA errors if not present

        detector = HeterodyneDetector(
            sample_rate=10e6,
            device=device
        )

        self.assertEqual(detector.device, device)
        # Check internal heterodyner device
        self.assertEqual(detector.heterodyner.device, device)

    def test_spatial_noise_canceller_device(self):
        """Verify SpatialNoiseCanceller respects the device parameter"""
        device = torch.device('cpu')

        # Need a geometry object
        geo = RadarGeometry(RADAR_GEOMETRY, device=device)

        canceller = SpatialNoiseCanceller(
            geometry=geo,
            config=NOISE_CANCELLATION,
            device=device
        )

        self.assertEqual(canceller.device, device)

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.get_device_name', return_value="Mock GPU")
    @patch('torch.cuda.get_device_properties')
    @patch('torch.randn')
    @patch('torch.matmul')
    def test_gpu_detection(self, mock_matmul, mock_randn, mock_props, mock_name, mock_available):
        """Verify main.py _setup_gpu logic (extracted)"""

        # We can't easily instantiate RadarApp because it does too much.
        # But we can import the class and test the method if we mock 'self'
        from main import RadarApp

        mock_props.return_value.total_memory = 8e9

        # Mock instance of RadarApp
        app = MagicMock()

        # Call _setup_gpu bound to the mock app
        # Since _setup_gpu is a method of RadarApp, we can access it via RadarApp._setup_gpu
        device, degraded = RadarApp._setup_gpu(app)

        self.assertEqual(device.type, 'cuda')
        self.assertEqual(device.index, 0)
        self.assertFalse(degraded)

    def test_radar_geometry_device(self):
        """Verify RadarGeometry creates tensors on the correct device"""
        device = torch.device('cpu')

        geo = RadarGeometry(RADAR_GEOMETRY, device=device)

        self.assertEqual(geo.tx1_pos.device, device)
        self.assertEqual(geo.rx1_pos.device, device)

        # Also check computation
        theta = torch.tensor([0.0], device=device)
        sv = geo.compute_steering_vector(theta)
        self.assertEqual(sv.device, device)

if __name__ == '__main__':
    unittest.main()
