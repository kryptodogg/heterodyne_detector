#!/usr/bin/env python3
"""
Advanced Pattern Matching for Heterodyne Detection
Uses DTW (Dynamic Time Warping) and Levenshtein distance for detecting
repeated phantom voice patterns and noise signatures
GPU-accelerated where possible
"""

import numpy as np
from collections import deque
import time

# DTW imports
try:
    from dtaidistance import dtw
    from dtaidistance import dtw_ndim
    DTW_AVAILABLE = True
except ImportError:
    print("Warning: dtaidistance not available. Install for DTW support.")
    DTW_AVAILABLE = False

# Heavy lifting is handled via Torch
cp = np
GPU_AVAILABLE = False


class PatternMatcher:
    def __init__(self, window_size=1024, overlap=0.5, max_patterns=1000, 
                 similarity_threshold=0.8, device='cpu', **kwargs):
        self.device = device
        """
        Initialize pattern matcher
        
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
        
        # Pattern storage
        self.known_patterns = deque(maxlen=max_patterns)
        self.pattern_metadata = deque(maxlen=max_patterns)
        
        # Statistics
        self.matches_found = 0
        self.total_comparisons = 0
        
    def extract_features(self, signal):
        """
        Extract features from signal for pattern matching
        Returns feature vectors suitable for DTW/Levenshtein
        """
        # Compute spectrogram
        from scipy import signal as sp_signal
        
        # STFT for time-frequency representation
        f, t, Zxx = sp_signal.stft(signal, nperseg=256, noverlap=128)
        
        # Convert to magnitude
        mag = np.abs(Zxx)
        
        # Normalize
        mag = mag / (np.max(mag) + 1e-10)
        
        # Extract spectral features
        features = {
            'spectrogram': mag,
            'spectral_centroid': self.compute_spectral_centroid(mag, f),
            'spectral_rolloff': self.compute_spectral_rolloff(mag, f),
            'zero_crossing_rate': self.compute_zero_crossings(signal),
            'energy_envelope': self.compute_energy_envelope(signal)
        }
        
        return features
    
    def compute_spectral_centroid(self, spectrogram, frequencies):
        """Compute spectral centroid over time"""
        weighted_sum = np.sum(spectrogram * frequencies[:, np.newaxis], axis=0)
        total_sum = np.sum(spectrogram, axis=0) + 1e-10
        return weighted_sum / total_sum
    
    def compute_spectral_rolloff(self, spectrogram, frequencies, rolloff=0.85):
        """Compute spectral rolloff over time"""
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
        """Compute zero crossing rate in windows"""
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
        """Create overlapping windows from signal"""
        windows = []
        for i in range(0, len(signal) - self.window_size + 1, self.hop_size):
            windows.append(signal[i:i+self.window_size])
        return windows
    
    def dtw_distance(self, series1, series2):
        """
        Compute DTW distance between two time series
        GPU accelerated if available
        """
        if not DTW_AVAILABLE:
            # Fallback to simple Euclidean distance
            min_len = min(len(series1), len(series2))
            return np.linalg.norm(series1[:min_len] - series2[:min_len])
        
        # Normalize series
        series1 = (series1 - np.mean(series1)) / (np.std(series1) + 1e-10)
        series2 = (series2 - np.mean(series2)) / (np.std(series2) + 1e-10)
        
        # Compute DTW distance with window constraint for speed
        window_size = int(0.1 * max(len(series1), len(series2)))
        distance = dtw.distance(series1, series2, window=window_size)
        
        return distance
    
    def spectral_dtw_distance(self, features1, features2):
        """
        Compute DTW distance using multiple spectral features
        Returns normalized distance (0-1)
        """
        distances = []
        
        # Compare spectral centroids
        if 'spectral_centroid' in features1 and 'spectral_centroid' in features2:
            d = self.dtw_distance(features1['spectral_centroid'], 
                                 features2['spectral_centroid'])
            distances.append(d)
        
        # Compare spectral rolloff
        if 'spectral_rolloff' in features1 and 'spectral_rolloff' in features2:
            d = self.dtw_distance(features1['spectral_rolloff'], 
                                 features2['spectral_rolloff'])
            distances.append(d)
        
        # Compare energy envelope
        if 'energy_envelope' in features1 and 'energy_envelope' in features2:
            d = self.dtw_distance(features1['energy_envelope'], 
                                 features2['energy_envelope'])
            distances.append(d)
        
        # Compare zero crossing rate
        if 'zero_crossing_rate' in features1 and 'zero_crossing_rate' in features2:
            d = self.dtw_distance(features1['zero_crossing_rate'], 
                                 features2['zero_crossing_rate'])
            distances.append(d)
        
        if not distances:
            return 1.0
        
        # Normalize and combine
        avg_distance = np.mean(distances)
        max_distance = np.max(distances) + 1e-10
        
        return avg_distance / max_distance
    
    def quantize_features(self, features, num_bins=8):
        """
        Quantize features into discrete symbols for Levenshtein distance
        Converts continuous features to strings for edit distance comparison
        """
        symbols = []
        
        # Quantize spectral centroid
        if 'spectral_centroid' in features:
            centroid = features['spectral_centroid']
            bins = np.linspace(np.min(centroid), np.max(centroid), num_bins)
            quantized = np.digitize(centroid, bins)
            symbols.append(''.join([chr(ord('A') + q) for q in quantized]))
        
        # Quantize energy envelope
        if 'energy_envelope' in features:
            energy = features['energy_envelope']
            bins = np.linspace(np.min(energy), np.max(energy), num_bins)
            quantized = np.digitize(energy, bins)
            symbols.append(''.join([chr(ord('A') + q) for q in quantized]))
        
        return '|'.join(symbols)
    
    def levenshtein_similarity(self, features1, features2):
        """
        Compute Levenshtein similarity between quantized features
        Returns similarity score (0-1)
        """
        if not LEVENSHTEIN_AVAILABLE:
            # Fallback to simple comparison
            return 0.5
        
        # Quantize features to strings
        str1 = self.quantize_features(features1)
        str2 = self.quantize_features(features2)
        
        if not str1 or not str2:
            return 0.0
        
        # Compute Levenshtein distance
        distance = Levenshtein.distance(str1, str2)
        
        # Normalize to similarity (0-1)
        max_len = max(len(str1), len(str2))
        similarity = 1.0 - (distance / max_len) if max_len > 0 else 0.0
        
        return similarity
    
    def compute_similarity(self, features1, features2, use_dtw=True, use_levenshtein=True):
        """
        Compute overall similarity between two feature sets
        Combines DTW and Levenshtein approaches
        """
        similarities = []
        
        if use_dtw and DTW_AVAILABLE:
            dtw_dist = self.spectral_dtw_distance(features1, features2)
            dtw_sim = 1.0 - dtw_dist
            similarities.append(dtw_sim)
        
        if use_levenshtein and LEVENSHTEIN_AVAILABLE:
            lev_sim = self.levenshtein_similarity(features1, features2)
            similarities.append(lev_sim)
        
        if not similarities:
            # Fallback to simple correlation
            if 'energy_envelope' in features1 and 'energy_envelope' in features2:
                e1 = features1['energy_envelope']
                e2 = features2['energy_envelope']
                min_len = min(len(e1), len(e2))
                corr = np.corrcoef(e1[:min_len], e2[:min_len])[0, 1]
                return max(0, corr)  # Clip negative correlations
            return 0.0
        
        # Weighted average (DTW more reliable for signals)
        if len(similarities) == 2:
            return 0.7 * similarities[0] + 0.3 * similarities[1]
        else:
            return np.mean(similarities)
    
    def find_matches(self, signal, max_matches=5):
        """
        Find matching patterns in stored patterns
        Returns list of matches with similarity scores
        """
        if len(self.known_patterns) == 0:
            return []
        
        # Extract features from input signal
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
        
        # Update statistics
        if matches:
            self.matches_found += 1
        
        return matches[:max_matches]
    
    def add_pattern(self, signal, metadata=None):
        """
        Add a new pattern to the known patterns database
        """
        features = self.extract_features(signal)
        
        # Check if pattern is too similar to existing ones
        for known_features in self.known_patterns:
            similarity = self.compute_similarity(features, known_features)
            if similarity > 0.95:
                # Pattern already exists
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
        """Clear all stored patterns"""
        self.known_patterns.clear()
        self.pattern_metadata.clear()
        self.matches_found = 0
        self.total_comparisons = 0
    
    def get_statistics(self):
        """Get matching statistics"""
        match_rate = (self.matches_found / self.total_comparisons * 100 
                     if self.total_comparisons > 0 else 0)
        
        return {
            'total_patterns': len(self.known_patterns),
            'total_comparisons': self.total_comparisons,
            'matches_found': self.matches_found,
            'match_rate': match_rate
        }


class PhantomVoiceDetector:
    """
    Specialized detector for phantom voice phenomena
    Uses pattern matching to identify repeated voice-like artifacts
    """
    
    def __init__(self, sample_rate=2.4e6):
        self.sample_rate = sample_rate
        self.pattern_matcher = PatternMatcher(
            window_size=4096,
            similarity_threshold=0.80
        )
        
        # Voice detection parameters
        self.voice_freq_range = (300, 3400)  # Typical voice range
        self.formant_frequencies = [(800, 1200), (1200, 2400), (2400, 3600)]
        
    def preprocess_for_voice(self, signal):
        """
        Preprocess signal for voice detection
        Includes demodulation and bandpass filtering
        """
        from scipy import signal as sp_signal
        
        # AM demodulation
        analytic_signal = sp_signal.hilbert(np.real(signal))
        envelope = np.abs(analytic_signal)
        
        # Downsample to audio rate
        audio_rate = 16000
        downsample_factor = int(self.sample_rate / audio_rate)
        
        if downsample_factor > 1:
            envelope = sp_signal.decimate(envelope, downsample_factor, zero_phase=True)
        
        # Bandpass filter for voice range
        nyquist = audio_rate / 2
        low = self.voice_freq_range[0] / nyquist
        high = self.voice_freq_range[1] / nyquist
        
        if high >= 1.0:
            high = 0.99
        
        b, a = sp_signal.butter(4, [low, high], btype='band')
        voice_signal = sp_signal.filtfilt(b, a, envelope)
        
        return voice_signal, audio_rate
    
    def detect_formants(self, signal, sample_rate):
        """
        Detect voice formants in signal
        Returns True if formant structure suggests voice
        """
        from scipy import signal as sp_signal
        
        # Compute power spectral density
        f, psd = sp_signal.welch(signal, fs=sample_rate, nperseg=1024)
        
        # Look for peaks in formant regions
        formant_scores = []
        
        for f_low, f_high in self.formant_frequencies:
            # Find indices in this formant range
            idx = np.where((f >= f_low) & (f <= f_high))[0]
            
            if len(idx) > 0:
                # Check for peak in this region
                formant_power = np.max(psd[idx])
                background_power = np.median(psd)
                
                # Formant should be significantly above background
                if background_power > 0:
                    formant_scores.append(formant_power / background_power)
        
        # Voice-like if multiple formants detected
        return len(formant_scores) >= 2 and np.mean(formant_scores) > 3.0
    
    def analyze_signal(self, signal):
        """
        Analyze signal for phantom voice characteristics
        Returns detection results
        """
        # Preprocess
        voice_signal, audio_rate = self.preprocess_for_voice(signal)
        
        # Detect formants
        has_formants = self.detect_formants(voice_signal, audio_rate)
        
        # Find pattern matches
        matches = self.pattern_matcher.find_matches(voice_signal)
        
        # Compute voice likelihood
        voice_likelihood = 0.0
        
        if has_formants:
            voice_likelihood += 0.5
        
        if matches:
            # Repeated patterns suggest artifact
            max_similarity = max([m['similarity'] for m in matches])
            voice_likelihood += 0.5 * max_similarity
        
        result = {
            'voice_likelihood': voice_likelihood,
            'has_formants': has_formants,
            'pattern_matches': len(matches),
            'is_phantom': voice_likelihood > 0.6 and len(matches) > 0
        }
        
        # Add as pattern if significant
        if voice_likelihood > 0.7:
            self.pattern_matcher.add_pattern(voice_signal, 
                                            metadata={'voice_likelihood': voice_likelihood})
        
        return result


def test_pattern_matcher():
    """Test the pattern matcher with synthetic data"""
    print("Testing Pattern Matcher...")
    
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
    matcher = PatternMatcher()
    
    # Add first pattern
    matcher.add_pattern(signal1, metadata={'name': 'pattern1'})
    print(f"Added pattern 1")
    
    # Find matches
    matches = matcher.find_matches(signal2)
    print(f"\nSignal 2 vs Signal 1:")
    print(f"  Matches found: {len(matches)}")
    if matches:
        print(f"  Best similarity: {matches[0]['similarity']:.3f}")
    
    matches = matcher.find_matches(signal3)
    print(f"\nSignal 3 vs Signal 1:")
    print(f"  Matches found: {len(matches)}")
    if matches:
        print(f"  Best similarity: {matches[0]['similarity']:.3f}")
    
    print(f"\nStatistics: {matcher.get_statistics()}")


if __name__ == "__main__":
    test_pattern_matcher()
