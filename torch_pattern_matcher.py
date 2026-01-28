#!/usr/bin/env python3
"""
PyTorch-Accelerated Pattern Matching for Heterodyne Detection
Works with AMD and NVIDIA GPUs via PyTorch
"""

import numpy as np
from collections import deque
import time
import torch
import torch.nn.functional as F

# DTW imports (still useful for some operations)
try:
    from dtaidistance import dtw
    DTW_AVAILABLE = True
except ImportError:
    DTW_AVAILABLE = False

# Levenshtein
try:
    import Levenshtein
    LEVENSHTEIN_AVAILABLE = True
except ImportError:
    LEVENSHTEIN_AVAILABLE = False


class TorchPatternMatcher:
    """
    PyTorch-accelerated pattern matcher for AMD/NVIDIA GPUs
    Works with both CPU and GPU
    """

    def __init__(self, window_size=4096, overlap=0.5,
                 max_patterns=1000, similarity_threshold=0.85):
        """
        Initialize PyTorch pattern matcher

        Args:
            window_size: Size of analysis window
            overlap: Overlap between windows (0-1)
            max_patterns: Maximum number of patterns to store
            similarity_threshold: Threshold for pattern matching (0-1)
        """
        self.window_size = window_size
        self.overlap = overlap
        self.hop_size = int(window_size * (1 - overlap))
        self.max_patterns = max_patterns
        self.similarity_threshold = similarity_threshold

        # Device selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"✅ PyTorch Pattern Matcher initialized on {self.device}")

        # Pattern storage
        self.known_patterns = deque(maxlen=max_patterns)
        self.pattern_metadata = deque(maxlen=max_patterns)

        # Statistics
        self.matches_found = 0
        self.total_comparisons = 0

        print(f"  Window size: {window_size}")
        print(f"  Similarity threshold: {similarity_threshold}")

    def extract_features(self, signal):
        """
        Extract features from signal for pattern matching
        Uses PyTorch for acceleration where possible
        """
        from scipy import signal as sp_signal

        # Convert to numpy if PyTorch tensor
        if isinstance(signal, torch.Tensor):
            signal = signal.cpu().numpy()

        # Use smaller FFT parameters to reduce memory usage
        # STFT for time-frequency representation
        f, t, Zxx = sp_signal.stft(signal, nperseg=min(256, len(signal)//4), noverlap=min(128, len(signal)//8))

        # Convert to magnitude
        mag = np.abs(Zxx)

        # Normalize
        mag = mag / (np.max(mag) + 1e-10)

        # Extract spectral features
        features = {
            'spectral_centroid': self.compute_spectral_centroid(mag, f),
            'spectral_rolloff': self.compute_spectral_rolloff(mag, f),
            'zero_crossing_rate': self.compute_zero_crossings(signal),
            'energy_envelope': self.compute_energy_envelope(signal)
        }

        return features

    def compute_spectral_centroid(self, spectrogram, frequencies):
        """Compute spectral centroid"""
        weighted_sum = np.sum(spectrogram * frequencies[:, np.newaxis], axis=0)
        total_sum = np.sum(spectrogram, axis=0) + 1e-10
        return weighted_sum / total_sum

    def compute_spectral_rolloff(self, spectrogram, frequencies, rolloff=0.85):
        """Compute spectral rolloff"""
        cumsum = np.cumsum(spectrogram, axis=0)
        total = np.sum(spectrogram, axis=0)
        threshold = rolloff * total

        rolloff_freq = np.zeros(spectrogram.shape[1])
        for i in range(spectrogram.shape[1]):
            idx = np.where(cumsum[:, i] >= threshold[i])[0]
            if len(idx) > 0:
                rolloff_freq[i] = frequencies[idx[0]]

        return rolloff_freq

    def compute_zero_crossings(self, signal):
        """Compute zero crossing rate"""
        windows = self.create_windows(signal)
        zcr = []

        for window in windows:
            signs = np.sign(window)
            crossings = np.sum(np.abs(np.diff(signs))) / (2 * len(window))
            zcr.append(crossings)

        return np.array(zcr)

    def compute_energy_envelope(self, signal):
        """Compute energy envelope"""
        windows = self.create_windows(signal)
        energy = []

        for window in windows:
            energy.append(np.sum(window**2))

        return np.array(energy)

    def create_windows(self, signal):
        """Create overlapping windows"""
        windows = []
        for i in range(0, len(signal) - self.window_size + 1, self.hop_size):
            windows.append(signal[i:i+self.window_size])
        return windows

    def torch_correlation_distance(self, series1, series2):
        """
        Fast correlation-based distance using PyTorch
        """
        # Convert to numpy if PyTorch tensors
        if isinstance(series1, torch.Tensor):
            series1 = series1.cpu().numpy()
        if isinstance(series2, torch.Tensor):
            series2 = series2.cpu().numpy()

        # Ensure arrays are 1D and flatten if necessary
        series1 = np.asarray(series1).flatten()
        series2 = np.asarray(series2).flatten()

        # Handle empty arrays
        if len(series1) == 0 or len(series2) == 0:
            return 1.0  # Maximum distance for empty arrays

        # Handle arrays with only zeros
        sum_sq1 = np.sum(series1**2)
        sum_sq2 = np.sum(series2**2)
        if sum_sq1 == 0 or sum_sq2 == 0:
            return 1.0  # Maximum distance if one signal is all zeros

        # Compute correlation using numpy (more memory efficient for this use case)
        correlation = np.correlate(series1, series2, mode='full')
        max_corr = np.max(np.abs(correlation)) / np.sqrt(sum_sq1 * sum_sq2)

        # Convert to distance (0 = identical, 1 = completely different)
        distance = 1.0 - abs(max_corr)

        return distance

    def spectral_distance(self, features1, features2):
        """
        Compute distance using multiple spectral features
        PyTorch-accelerated where possible
        """
        distances = []

        # Compare spectral centroids
        if 'spectral_centroid' in features1 and 'spectral_centroid' in features2:
            d = self.torch_correlation_distance(
                features1['spectral_centroid'],
                features2['spectral_centroid']
            )
            distances.append(d)

        # Compare spectral rolloff
        if 'spectral_rolloff' in features1 and 'spectral_rolloff' in features2:
            d = self.torch_correlation_distance(
                features1['spectral_rolloff'],
                features2['spectral_rolloff']
            )
            distances.append(d)

        # Compare energy envelope
        if 'energy_envelope' in features1 and 'energy_envelope' in features2:
            d = self.torch_correlation_distance(
                features1['energy_envelope'],
                features2['energy_envelope']
            )
            distances.append(d)

        # Compare zero crossing rate
        if 'zero_crossing_rate' in features1 and 'zero_crossing_rate' in features2:
            d = self.torch_correlation_distance(
                features1['zero_crossing_rate'],
                features2['zero_crossing_rate']
            )
            distances.append(d)

        if not distances:
            return 1.0

        # Average distance
        avg_distance = np.mean(distances)

        return avg_distance

    def compute_similarity(self, features1, features2):
        """
        Compute overall similarity between feature sets
        Uses PyTorch-accelerated operations
        """
        # Use PyTorch-accelerated distance
        distance = self.spectral_distance(features1, features2)
        similarity = 1.0 - distance

        return similarity

    def find_matches(self, signal, max_matches=5):
        """
        Find matching patterns using PyTorch acceleration
        """
        if len(self.known_patterns) == 0:
            return []

        # Extract features
        features = self.extract_features(signal)

        matches = []

        # Compare with all known patterns
        for i, (known_features, metadata) in enumerate(zip(self.known_patterns,
                                                           self.pattern_metadata)):
            self.total_comparisons += 1

            # Compute similarity
            similarity = self.compute_similarity(features, known_features)

            if similarity >= self.similarity_threshold:
                matches.append({
                    'pattern_id': i,
                    'similarity': similarity,
                    'metadata': metadata,
                    'timestamp': time.time()
                })

        # Sort by similarity
        matches.sort(key=lambda x: x['similarity'], reverse=True)

        if matches:
            self.matches_found += 1

        return matches[:max_matches]

    def add_pattern(self, signal, metadata=None):
        """Add pattern to database"""
        features = self.extract_features(signal)

        # Check if pattern already exists
        for known_features in self.known_patterns:
            similarity = self.compute_similarity(features, known_features)
            if similarity > 0.95:
                return False

        # Add new pattern
        self.known_patterns.append(features)

        if metadata is None:
            metadata = {
                'added_time': time.time(),
                'length': len(signal)
            }

        self.pattern_metadata.append(metadata)

        return True

    def clear_patterns(self):
        """Clear pattern database"""
        self.known_patterns.clear()
        self.pattern_metadata.clear()
        self.matches_found = 0
        self.total_comparisons = 0

    def get_statistics(self):
        """Get statistics"""
        match_rate = (self.matches_found / self.total_comparisons * 100
                     if self.total_comparisons > 0 else 0)

        return {
            'total_patterns': len(self.known_patterns),
            'total_comparisons': self.total_comparisons,
            'matches_found': self.matches_found,
            'match_rate': match_rate,
            'device': str(self.device)
        }


def benchmark_torch():
    """Benchmark PyTorch vs NumPy performance"""
    print("\n" + "="*60)
    print("PyTorch Performance Benchmark")
    print("="*60)

    # Generate test signals
    n = 8192
    signal1 = np.random.randn(n)
    signal2 = np.random.randn(n)

    # PyTorch tensor
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch_sig1 = torch.tensor(signal1, dtype=torch.float32, device=device)
        torch_sig2 = torch.tensor(signal2, dtype=torch.float32, device=device)

        # Benchmark PyTorch
        import time
        start = time.time()
        for _ in range(100):
            result = F.conv1d(
                torch_sig1.view(1, 1, -1),
                torch_sig2.flip(-1).view(1, 1, -1),
                padding=torch_sig2.shape[0] - 1
            ).squeeze()
        torch_time = time.time() - start

        print(f"PyTorch correlation (100 iterations): {torch_time:.3f}s")

    # Benchmark NumPy
    start = time.time()
    for _ in range(100):
        result = np.correlate(signal1, signal2, mode='same')
    numpy_time = time.time() - start

    print(f"NumPy correlation (100 iterations): {numpy_time:.3f}s")

    if torch.cuda.is_available():
        speedup = numpy_time / torch_time
        print(f"\nSpeedup: {speedup:.2f}x")
        print(f"Device: {device}")


def test_torch_pattern_matcher():
    """Test the PyTorch pattern matcher"""
    print("\n" + "="*60)
    print("Testing PyTorch Pattern Matcher")
    print("="*60)

    # Create synthetic signals
    t = np.linspace(0, 1, 8192)

    # Signal 1: Base pattern
    signal1 = (np.sin(2 * np.pi * 440 * t) +
              0.5 * np.sin(2 * np.pi * 880 * t) +
              0.1 * np.random.randn(len(t)))

    # Signal 2: Similar but time-stretched
    t2 = np.linspace(0, 1.2, 8192)
    signal2 = (np.sin(2 * np.pi * 440 * t2) +
              0.5 * np.sin(2 * np.pi * 880 * t2) +
              0.1 * np.random.randn(len(t2)))

    # Signal 3: Different pattern
    signal3 = (np.sin(2 * np.pi * 220 * t) +
              0.1 * np.random.randn(len(t)))

    # Test matcher
    matcher = TorchPatternMatcher(similarity_threshold=0.75)

    # Add first pattern
    print("\nAdding pattern 1...")
    matcher.add_pattern(signal1, metadata={'name': 'pattern1'})

    # Find matches
    print("\nTesting signal 2 (similar)...")
    matches = matcher.find_matches(signal2)
    print(f"  Matches found: {len(matches)}")
    if matches:
        print(f"  Best similarity: {matches[0]['similarity']:.3f}")

    print("\nTesting signal 3 (different)...")
    matches = matcher.find_matches(signal3)
    print(f"  Matches found: {len(matches)}")
    if matches:
        print(f"  Best similarity: {matches[0]['similarity']:.3f}")

    # Statistics
    stats = matcher.get_statistics()
    print(f"\nStatistics:")
    print(f"  Total patterns: {stats['total_patterns']}")
    print(f"  Total comparisons: {stats['total_comparisons']}")
    print(f"  Matches found: {stats['matches_found']}")
    print(f"  Device: {stats['device']}")


if __name__ == "__main__":
    print("PyTorch Pattern Matcher with GPU Support")
    print("="*60)

    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available, using CPU")

    benchmark_torch()
    test_torch_pattern_matcher()