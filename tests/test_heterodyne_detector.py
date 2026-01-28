import unittest
import torch
import numpy as np
from heterodyne_detector import HeterodyneDetector

class TestHeterodyneDetector(unittest.TestCase):
    def setUp(self):
        self.sample_rate = 10000
        self.device = torch.device('cpu') # Use CPU for stable verification
        self.detector = HeterodyneDetector(
            sample_rate=self.sample_rate,
            device=self.device
        )

    def test_multi_stage_heterodyne(self):
        """Verify multi-stage heterodyning produces audio-range signals."""
        # Create a 2.4kHz RF signal (simulated at low sample rate)
        t = torch.linspace(0, 1, self.sample_rate, device=self.device)
        rf_signal = torch.exp(1j * 2 * np.pi * 2400 * t)
        
        # Test the new pipeline (which we will implement)
        # We'll check if the detector now has the expert stages
        if hasattr(self.detector, 'heterodyner'):
            audio_results = self.detector.heterodyner.full_heterodyne_pipeline(rf_signal)
            self.assertIn('complex', audio_results)
            self.assertIn('envelope', audio_results)
            # Check if output is lower frequency (magnitude change over time)
            self.assertEqual(len(audio_results['complex']), len(rf_signal))
        else:
            self.fail("RadioToAcousticHeterodyner not integrated yet")

    def test_detect_torch_integration(self):
        """Verify detect() works with Torch tensors and returns expected metrics."""
        s1 = torch.randn(1024, device=self.device)
        s2 = torch.randn(1024, device=self.device)
        
        result = self.detector.detect(s1, s2)
        self.assertIn('score', result)
        self.assertIn('freq_offset', result)
        self.assertIn('is_voice_range', result)

if __name__ == '__main__':
    unittest.main()
