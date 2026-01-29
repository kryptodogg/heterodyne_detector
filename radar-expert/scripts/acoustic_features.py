#!/usr/bin/env python3
"""
Acoustic Feature Extraction from Heterodyned Radar Signals

Extracts meaningful features from audio-range signals for:
- Activity classification (presence, motion type)
- Biometric identification (gait, heartbeat)
- Speech characteristics (formants, pitch)
- Signal quality metrics
"""

import numpy as np
import torch
import scipy.signal as signal
from scipy.fft import fft, fftfreq


class AcousticFeatureExtractor:
    """
    Extract perceptually-relevant features from audio-range heterodyned signals
    """

    def __init__(self, sample_rate_hz, device='cpu'):
        self.sample_rate = sample_rate_hz
        self.device = device
        self.nyquist = sample_rate_hz / 2

    def extract_mfcc_features(self, audio_signal, n_mfcc=13, n_fft=512):
        """
        Mel-Frequency Cepstral Coefficients (MFCCs)

        Standard speech recognition feature set, highly relevant for
        heterodyned speech-like signals

        Args:
            audio_signal: Real or complex audio signal
            n_mfcc: Number of MFCC coefficients
            n_fft: FFT size

        Returns:
            mfcc_features: (n_mfcc, n_frames)
        """

        # Convert to real if complex
        if torch.is_complex(audio_signal):
            audio_real = torch.abs(audio_signal)
        else:
            audio_real = audio_signal

        # Compute magnitude spectrogram
        n_hop = n_fft // 2
        stft_matrix = torch.stft(
            audio_real,
            n_fft=n_fft,
            hop_length=n_hop,
            window=torch.hann_window(n_fft, device=self.device),
            return_complex=True
        )
        mag_spec = torch.abs(stft_matrix)  # (n_fft//2 + 1, n_frames)

        # Mel filterbank
        mel_filters = self._create_mel_filterbank(n_fft, n_mfcc)

        # Apply mel filters
        mel_spec = torch.matmul(mel_filters, mag_spec)

        # Log scaling
        mel_spec_db = 20 * torch.log10(mel_spec + 1e-9)

        # DCT (Discrete Cosine Transform) to get MFCCs
        mfcc = torch.nn.functional.linear(
            mel_spec_db.T,  # (n_frames, n_mfcc)
            self._dct_matrix(n_mfcc).T.to(self.device)
        ).T  # (n_mfcc, n_frames)

        return mfcc

    def _create_mel_filterbank(self, n_fft, n_mfcc):
        """Create mel-scaled triangular filters"""

        # Mel scale conversion
        mel_min = 2595 * np.log10(1 + 20 / 700)
        mel_max = 2595 * np.log10(1 + self.nyquist / 700)

        mel_points = np.linspace(mel_min, mel_max, n_mfcc + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)

        # Convert to FFT bin indices
        bin_points = np.floor((n_fft + 1) * hz_points / self.sample_rate).astype(np.int32)

        # Create triangular filters
        filters = np.zeros((n_mfcc, n_fft // 2 + 1))

        for m in range(1, n_mfcc + 1):
            f_m_minus = bin_points[m - 1]
            f_m = bin_points[m]
            f_m_plus = bin_points[m + 1]

            if f_m > f_m_minus:
                filters[m - 1, f_m_minus:f_m] = np.linspace(0, 1, f_m - f_m_minus)
            if f_m_plus > f_m:
                filters[m - 1, f_m:f_m_plus] = np.linspace(1, 0, f_m_plus - f_m)

        return torch.tensor(filters, dtype=torch.float32, device=self.device)

    def _dct_matrix(self, n):
        """Create DCT-II matrix"""
        dct_mat = np.zeros((n, n))
        for k in range(n):
            for m in range(n):
                if k == 0:
                    c_k = np.sqrt(1 / n)
                else:
                    c_k = np.sqrt(2 / n)
                dct_mat[k, m] = c_k * np.cos(np.pi * k * (2 * m + 1) / (2 * n))
        return torch.tensor(dct_mat, dtype=torch.float32, device=self.device)

    def extract_pitch_contour(self, audio_signal, fmin_hz=80, fmax_hz=400):
        """
        Extract fundamental frequency (pitch) contour over time

        Args:
            audio_signal: Audio signal
            fmin_hz: Minimum pitch frequency
            fmax_hz: Maximum pitch frequency

        Returns:
            pitch_contour: Pitch in Hz at each frame
            confidence: Pitch confidence (0-1)
        """

        if torch.is_complex(audio_signal):
            audio_real = torch.abs(audio_signal)
        else:
            audio_real = audio_signal

        # Autocorrelation-based pitch detection
        n_fft = 4096
        n_hop = n_fft // 4

        pitch_contour = []
        confidence_vals = []

        for start in range(0, len(audio_real) - n_fft, n_hop):
            frame = audio_real[start:start + n_fft]

            # Autocorrelation
            acf = np.correlate(frame.cpu().numpy(), frame.cpu().numpy(), mode='full')
            acf = acf[len(acf) // 2:]

            # Find peaks in pitch range
            min_lag = int(self.sample_rate / fmax_hz)
            max_lag = int(self.sample_rate / fmin_hz)

            peaks, properties = signal.find_peaks(
                acf[min_lag:max_lag],
                height=np.max(acf) * 0.3
            )

            if len(peaks) > 0:
                # Strongest peak = fundamental
                peak_lag = peaks[np.argmax(properties['peak_heights'])] + min_lag
                pitch = self.sample_rate / peak_lag
                confidence = properties['peak_heights'][np.argmax(properties['peak_heights'])] / acf[0]
            else:
                pitch = 0
                confidence = 0

            pitch_contour.append(pitch)
            confidence_vals.append(confidence)

        return np.array(pitch_contour), np.array(confidence_vals)

    def extract_doppler_modulation(self, iq_signal):
        """
        Extract Doppler-induced frequency modulation

        Detects motion patterns (velocity changes, micro-vibrations)

        Args:
            iq_signal: Complex IQ signal

        Returns:
            doppler_envelope: Instantaneous frequency modulation
            velocity_profile: Estimated velocity trajectory
        """

        # Instantaneous phase
        phase = torch.angle(iq_signal)

        # Phase unwrapping
        phase_diff = torch.diff(phase, prepend=phase[0:1])
        phase_unwrapped = torch.cumsum(
            torch.where(
                torch.abs(phase_diff) > np.pi,
                phase_diff - 2 * np.pi,
                phase_diff
            ),
            dim=0
        )

        # Instantaneous frequency (Doppler shift)
        inst_freq = phase_unwrapped * self.sample_rate / (2 * np.pi)

        # Velocity (assuming 2.4 GHz carrier)
        wavelength = 3e8 / 2.4e9  # ~0.125 m
        velocity = inst_freq * wavelength / 2  # Radar Doppler relation

        return inst_freq, velocity

    def extract_activity_signature(self, audio_signal, window_size_ms=100):
        """
        Extract activity classification features

        Distinguishes: stationary, walking, running, vibration, etc.

        Args:
            audio_signal: Audio signal
            window_size_ms: Analysis window

        Returns:
            activity_features: Dict with classification metrics
        """

        if torch.is_complex(audio_signal):
            audio_real = torch.abs(audio_signal)
        else:
            audio_real = audio_signal

        window_samples = int(window_size_ms * self.sample_rate / 1000)

        # Segment signal
        features = {
            'rms_energy': [],
            'zero_crossing_rate': [],
            'spectral_centroid': [],
            'spectral_spread': [],
            'peak_frequencies': []
        }

        n_frames = len(audio_real) // window_samples

        for frame_idx in range(n_frames):
            start = frame_idx * window_samples
            frame = audio_real[start:start + window_samples]

            # RMS Energy
            rms = torch.sqrt(torch.mean(frame ** 2))
            features['rms_energy'].append(float(rms))

            # Zero Crossing Rate
            zcr = torch.sum(torch.abs(torch.sign(frame[1:]) - torch.sign(frame[:-1]))) / (2 * len(frame))
            features['zero_crossing_rate'].append(float(zcr))

            # Spectral features
            spec = torch.abs(torch.fft.rfft(frame))
            freqs = torch.fft.rfftfreq(len(frame), 1.0 / self.sample_rate)

            # Spectral centroid (center of mass)
            sc = torch.sum(freqs * spec) / torch.sum(spec)
            features['spectral_centroid'].append(float(sc))

            # Spectral spread (variance)
            ss = torch.sqrt(torch.sum((freqs - sc) ** 2 * spec) / torch.sum(spec))
            features['spectral_spread'].append(float(ss))

            # Peak frequency
            peak_freq = freqs[torch.argmax(spec)]
            features['peak_frequencies'].append(float(peak_freq))

        # Convert to tensors
        for key in features:
            features[key] = torch.tensor(features[key], device=self.device)

        return features

    def extract_voice_quality_metrics(self, audio_signal, reference_formants=None):
        """
        Estimate speech-likeness and quality

        Returns metrics useful for heterodyned speech detection

        Args:
            audio_signal: Audio signal
            reference_formants: Known human formants for comparison

        Returns:
            quality_metrics: Dict with voice-likeness scores
        """

        if torch.is_complex(audio_signal):
            audio_real = torch.abs(audio_signal)
        else:
            audio_real = audio_signal

        # Default human formants
        if reference_formants is None:
            reference_formants = [700, 1220, 2600]  # Hz (male)

        # Compute spectrum
        spec = torch.abs(torch.fft.rfft(audio_real, n=4096))
        freqs = torch.fft.rfftfreq(4096, 1.0 / self.sample_rate)

        # Spectral centroid (should be in voice range ~1-3 kHz)
        spec_centroid = torch.sum(freqs * spec) / torch.sum(spec)

        # Formant-likeness: peak power near expected formants
        formant_power = 0
        for f_form in reference_formants:
            formant_bin = int(f_form / (self.sample_rate / 4096))
            if formant_bin < len(spec):
                formant_power += float(spec[formant_bin])

        total_power = torch.sum(spec)
        formant_ratio = formant_power / (float(total_power) + 1e-9)

        # Voicing strength (ratio of low-freq to high-freq)
        low_freq_power = torch.sum(spec[freqs < 1000])
        high_freq_power = torch.sum(spec[freqs > 3000])
        voicing_strength = low_freq_power / (high_freq_power + 1e-9)

        return {
            'spectral_centroid_hz': float(spec_centroid),
            'formant_ratio': float(formant_ratio),
            'voicing_strength': float(voicing_strength),
            'voice_likeness_score': float(formant_ratio * voicing_strength),  # Combined metric
            'cepstral_peak_prominence': float(torch.max(spec) / torch.mean(spec))
        }


class MultiFrequencyAnalyzer:
    """
    Analyze heterodyned signals across multiple frequency bands simultaneously
    """

    def __init__(self, sample_rate_hz, num_bands=3, device='cpu'):
        self.sample_rate = sample_rate_hz
        self.num_bands = num_bands
        self.device = device

        # Pre-defined frequency bands (for different radar types)
        self.band_configs = {
            'isb_bands': {  # Industrial, Scientific, Medical
                '2.4GHz': (2.4e9, 100e3),
                '5.8GHz': (5.8e9, 100e3),
                '24GHz': (24e9, 1e6)
            },
            'audio_bands': {  # Heterodyned audio output
                'bass': (100, 500),
                'mid': (500, 2000),
                'treble': (2000, 5000)
            }
        }

    def analyze_multi_band_signal(self, iq_signal, band_set='audio_bands'):
        """
        Analyze signal in multiple frequency bands simultaneously

        Args:
            iq_signal: Input IQ signal
            band_set: 'isb_bands' or 'audio_bands'

        Returns:
            multi_band_features: Dict of features per band
        """

        bands = self.band_configs[band_set]
        features = {}

        for band_name, (center_freq, bandwidth) in bands.items():
            # Design band-pass filter
            if bandwidth < self.sample_rate / 4:
                low_freq = center_freq - bandwidth / 2
                high_freq = center_freq + bandwidth / 2

                # Normalized frequencies
                low_norm = max(0.01, min(0.99, 2 * low_freq / self.sample_rate))
                high_norm = max(0.01, min(0.99, 2 * high_freq / self.sample_rate))

                # Filter
                b, a = signal.butter(4, [low_norm, high_norm], btype='band')
                filtered_np = signal.filtfilt(b, a, iq_signal.cpu().numpy())
                filtered = torch.tensor(filtered_np, dtype=iq_signal.dtype, device=self.device)
            else:
                filtered = iq_signal

            # Extract features from this band
            band_power = torch.mean(torch.abs(filtered) ** 2)
            band_snr = torch.max(torch.abs(filtered)) / (torch.mean(torch.abs(filtered)) + 1e-9)

            features[band_name] = {
                'power_db': float(10 * torch.log10(band_power + 1e-9)),
                'snr_db': float(10 * torch.log10(band_snr + 1e-9)),
                'peak_magnitude': float(torch.max(torch.abs(filtered)))
            }

        return features


if __name__ == "__main__":
    print("Acoustic Feature Extraction")
    print("=" * 70)

    sample_rate = 48000  # 48 kHz audio
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create test signal
    duration = 1.0
    t = np.arange(int(duration * sample_rate)) / sample_rate

    # Simulated heterodyned audio (formants + pitch contour)
    f1, f2, f3 = 700, 1220, 2600  # Formants
    audio = 0.3 * (np.sin(2*np.pi*f1*t) + 0.5*np.sin(2*np.pi*f2*t) + 0.3*np.sin(2*np.pi*f3*t))
    audio = torch.tensor(audio, dtype=torch.float32, device=device)

    # Extract features
    extractor = AcousticFeatureExtractor(sample_rate, device=device)

    print("Extracting MFCCs...")
    mfccs = extractor.extract_mfcc_features(audio, n_mfcc=13)
    print(f"  Shape: {mfccs.shape}")

    print("\nExtracting pitch contour...")
    pitch, confidence = extractor.extract_pitch_contour(audio)
    print(f"  Mean pitch: {np.mean(pitch[confidence > 0.3]):.1f} Hz")

    print("\nExtracting activity signature...")
    activity = extractor.extract_activity_signature(audio, window_size_ms=50)
    print(f"  Mean RMS energy: {np.mean(activity['rms_energy']):.3f}")
    print(f"  Mean zero crossing rate: {np.mean(activity['zero_crossing_rate']):.3f}")

    print("\nExtracting voice quality...")
    quality = extractor.extract_voice_quality_metrics(audio)
    print(f"  Spectral centroid: {quality['spectral_centroid_hz']:.0f} Hz")
    print(f"  Voice-likeness score: {quality['voice_likeness_score']:.3f}")
