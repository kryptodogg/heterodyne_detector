import unittest
import torch
import numpy as np
from noise_canceller import SpatialNoiseCanceller

class TestNoiseCanceller(unittest.TestCase):
    def setUp(self):
        # Mock geometry
        class MockGeometry:
            def compute_steering_vector(self, angles):
                return torch.ones((len(angles), 4), dtype=torch.complex64)
        
        self.geometry = MockGeometry()
        self.config = {
            'num_steering_angles': 37,
            'filter_length': 64,
            'learning_rate': 0.01
        }
        self.device = torch.device('cpu')
        self.canceller = SpatialNoiseCanceller(self.geometry, self.config, self.device)

    def test_notch_filter(self):
        """Verify frequency-domain notch filter removes tonal interference."""
        # Create a signal with a 60Hz tone
        fs = 1000
        t = torch.linspace(0, 1, fs, device=self.device)
        target = 0.5 * torch.sin(2 * np.pi * 10 * t) # 10Hz target
        tone = 0.3 * torch.sin(2 * np.pi * 60 * t) # 60Hz interference
        signal = target + tone
        
        # This method might not exist yet
        try:
            filtered = self.canceller.frequency_domain_notch_filter(signal, center_freq_hz=60, sample_rate=fs)
            
            # Check power reduction at 60Hz
            # Target 10Hz should be preserved
            self.assertEqual(len(filtered), len(signal))
            
            # Simple check: power of filtered should be less than original but more than target
            p_orig = torch.mean(signal**2)
            p_filt = torch.mean(filtered**2)
            p_target = torch.mean(target**2)
            
            self.assertLess(p_filt, p_orig)
            # Should be close to target power
            self.assertAlmostEqual(float(p_filt.item()), float(p_target.item()), delta=0.1)
            
        except AttributeError:
            self.fail("Method 'frequency_domain_notch_filter' not implemented")

if __name__ == '__main__':
    unittest.main()
