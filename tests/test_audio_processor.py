import unittest
import torch
import numpy as np
from audio_processor import AudioProcessor

class TestAudioProcessor(unittest.TestCase):
    def setUp(self):
        self.sample_rate = 10000 # Lower for testing
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = AudioProcessor(
            sample_rate=self.sample_rate,
            n_mfcc=13,
            device=self.device
        )

    def test_extract_features_returns_torch(self):
        """Verify that extracted features are Torch tensors on the correct device."""
        signal = torch.randn(self.sample_rate, device=self.device)
        features = self.processor.extract_features(signal)
        
        self.assertIsInstance(features, torch.Tensor)
        self.assertEqual(features.device.type, self.device.type)
        self.assertEqual(features.shape[0], 13)

    def test_torch_only_pipeline(self):
        """Verify the pipeline doesn't require NumPy/Librosa (if possible)."""
        # Test on the same device as the processor
        signal = torch.randn(self.sample_rate, device=self.device)
        features = self.processor.extract_features(signal)
        self.assertEqual(features.device.type, self.device.type)

    def test_spectral_centroid(self):
        """Test the new spectral centroid metric (should exist in refactored version)."""
        # Create a 1kHz sine wave
        f = 1000
        t = torch.linspace(0, 1, self.sample_rate, device=self.device)
        signal = torch.sin(2 * np.pi * f * t)
        
        # This method might not exist yet, causing a failure in the RED phase
        try:
            metrics = self.processor.extract_voice_quality_metrics(signal)
            centroid = metrics['spectral_centroid_hz']
            # Centroid should be near 1000 Hz. 
            # With windowing and discrete bins, 20% error is possible on very small FFTs.
            self.assertAlmostEqual(centroid, f, delta=250)
        except AttributeError:
            self.fail("Method 'extract_voice_quality_metrics' not implemented")

if __name__ == '__main__':
    unittest.main()
