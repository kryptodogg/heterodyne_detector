#!/usr/bin/env python3
"""
Active Audio Cancellation via Anti-Phase Signal Synthesis

Extends spatial noise cancellation to acoustic domain:
- Synthesizes anti-phase audio signals at perceived frequencies
- Cancels specific tones (e.g., power line hum, interference)
- Enables selective audio suppression (speech vs. background)
- Real-time adaptive cancellation

Theory: ANC (Active Noise Control) - generate inverted signal to destructively interfere
"""

import numpy as np
import torch
import scipy.signal as signal


class ActiveAudioCanceller:
    """
    Synthesize anti-phase audio signals for destructive interference

    Applications:
    - Suppress audible interference (60/50 Hz mains, harmonics)
    - Selective tone cancellation (siren, speech)
    - Privacy enhancement (cancel intelligible speech)
    """

    def __init__(self, sample_rate_hz, num_channels=1, device='cpu'):
        self.sample_rate = sample_rate_hz
        self.num_channels = num_channels
        self.device = device
        self.nyquist = sample_rate_hz / 2

    def estimate_target_frequency(self, audio_signal, fft_size=2048):
        """
        Identify dominant frequency for cancellation

        Args:
            audio_signal: Audio signal (real or complex)
            fft_size: FFT size for spectral analysis

        Returns:
            freq_hz: Identified frequency
            magnitude: Power at that frequency
            confidence: Estimate confidence (0-1)
        """

        if torch.is_complex(audio_signal):
            audio_real = torch.abs(audio_signal)
        else:
            audio_real = audio_signal

        # Compute spectrum
        spectrum = torch.abs(torch.fft.rfft(audio_real, n=fft_size))
        freqs = torch.fft.rfftfreq(fft_size, 1.0 / self.sample_rate)

        # Find peak
        peak_bin = torch.argmax(spectrum)
        peak_freq = float(freqs[peak_bin])
        peak_magnitude = float(spectrum[peak_bin])

        # Confidence: ratio of peak to mean
        confidence = peak_magnitude / (torch.mean(spectrum) + 1e-9)
        confidence = float(torch.tanh(torch.tensor(confidence / 10)))  # Normalize to [0-1]

        return peak_freq, peak_magnitude, confidence

    def synthesize_cancellation_signal(self, audio_signal, target_freq=None,
                                      phase_offset_deg=180):
        """
        Generate anti-phase signal for active cancellation

        Args:
            audio_signal: Input audio (to extract amplitude modulation)
            target_freq: Frequency to cancel (auto-detected if None)
            phase_offset_deg: Phase inversion (typically 180° for cancellation)

        Returns:
            cancellation_signal: Anti-phase audio
            cancellation_info: Diagnostics
        """

        if torch.is_complex(audio_signal):
            audio_real = torch.abs(audio_signal)
        else:
            audio_real = audio_signal

        # Estimate target frequency
        if target_freq is None:
            target_freq, _, _ = self.estimate_target_frequency(audio_signal)

        # Extract envelope (amplitude modulation of audio)
        # Use analytic signal to get envelope
        analytic = signal.hilbert(audio_real.cpu().numpy())
        envelope = np.abs(analytic)

        t = np.arange(len(audio_real)) / self.sample_rate
        phase_offset_rad = np.radians(phase_offset_deg)

        # Generate anti-phase carrier
        carrier = np.exp(1j * (2 * np.pi * target_freq * t + phase_offset_rad))

        # Modulate carrier by envelope
        cancellation = envelope * carrier

        # Convert to tensor
        cancellation = torch.tensor(cancellation, dtype=torch.complex64, device=self.device)

        # Estimate optimal amplitude scaling
        # Use cross-correlation to find best amplitude match
        cross_corr = np.correlate(audio_real.cpu().numpy(), np.abs(cancellation.cpu().numpy()),
                                 mode='valid')
        amplitude_scale = np.max(cross_corr) / (np.sum(np.abs(cancellation.cpu().numpy()) ** 2) + 1e-9)

        cancellation_scaled = amplitude_scale * cancellation

        return cancellation_scaled, {
            'target_freq_hz': target_freq,
            'amplitude_scale': float(amplitude_scale),
            'phase_offset_deg': phase_offset_deg
        }

    def adaptive_lms_cancellation(self, target_signal, reference_signal,
                                 filter_length=256, learning_rate=0.01, num_iters=None):
        """
        Adaptive LMS-based active cancellation

        Similar to RF domain LMS, but optimized for audio frequencies

        Args:
            target_signal: Signal to be cancelled
            reference_signal: Noise reference (correlated with noise to cancel)
            filter_length: Adaptive filter tap count
            learning_rate: LMS step size
            num_iters: Number of iterations (default: full length)

        Returns:
            cancelled_signal: Cleaned output
            filter_weights: Learned cancellation filter
            error_trace: Error over iterations
        """

        if num_iters is None:
            num_iters = len(target_signal)

        # Initialize weights
        w = torch.zeros(filter_length, dtype=torch.complex64, device=self.device)

        cancelled = torch.zeros(num_iters, dtype=torch.complex64, device=self.device)
        error = torch.zeros(num_iters, dtype=torch.complex64, device=self.device)

        for n in range(num_iters):
            # Reference signal window (causal)
            ref_start = max(0, n - filter_length)
            x_n = reference_signal[ref_start:n+1]

            # Pad to filter_length
            if len(x_n) < filter_length:
                x_n = torch.cat([
                    torch.zeros(filter_length - len(x_n), dtype=torch.complex64, device=self.device),
                    x_n
                ])

            # Estimate interference
            y_n = torch.dot(w.conj(), x_n)

            # Error (residual)
            err = target_signal[n] - y_n
            error[n] = err

            # Weight update
            w = w + learning_rate * err * x_n.conj()

            # Cancelled signal
            cancelled[n] = err

        return cancelled, w, error

    def frequency_domain_notch_filter(self, audio_signal, center_freq_hz, bandwidth_hz=10):
        """
        Design and apply frequency-domain notch (bandstop) filter

        Effective for tonal interference

        Args:
            audio_signal: Input audio
            center_freq_hz: Notch center frequency
            bandwidth_hz: Notch bandwidth (-3dB points)

        Returns:
            filtered_signal: Audio with notch applied
        """

        # FFT
        spectrum = torch.fft.fft(audio_signal)
        freqs = torch.fft.fftfreq(len(audio_signal), 1.0 / self.sample_rate)

        # Create notch mask
        freq_abs = torch.abs(freqs)
        notch_mask = 1.0 - torch.exp(
            -((freq_abs - center_freq_hz) ** 2) / (2 * (bandwidth_hz / 4) ** 2)
        )

        # Apply notch
        spectrum_notched = spectrum * notch_mask

        # IFFT
        filtered = torch.fft.ifft(spectrum_notched)

        return torch.real(filtered)

    def predict_cancellation_error(self, target_signal, cancellation_signal):
        """
        Estimate residual error after cancellation

        Useful for monitoring cancellation effectiveness

        Args:
            target_signal: Original signal
            cancellation_signal: Anti-phase signal to subtract

        Returns:
            residual_error: Remaining signal after cancellation
            cancellation_ratio_db: Suppression in dB
            cancellation_depth_percent: Percentage error reduction
        """

        # Ensure same length
        min_len = min(len(target_signal), len(cancellation_signal))
        target = target_signal[:min_len]
        cancel = cancellation_signal[:min_len]

        # Residual
        residual = target - cancel

        # Power before and after
        power_before = torch.mean(torch.abs(target) ** 2)
        power_after = torch.mean(torch.abs(residual) ** 2)

        # Cancellation metrics
        cancellation_ratio_db = 10 * torch.log10(power_before / (power_after + 1e-12))
        cancellation_depth = (1 - power_after / (power_before + 1e-12)) * 100

        return residual, float(cancellation_ratio_db), float(cancellation_depth)

    def generate_harmonic_cancellation(self, audio_signal, fundamental_freq_hz,
                                     num_harmonics=5):
        """
        Generate cancellation for fundamental + harmonics (for complex tones)

        Args:
            audio_signal: Input audio
            fundamental_freq_hz: Base frequency
            num_harmonics: Number of harmonics to cancel

        Returns:
            cancellation_signal: Combined anti-phase for all harmonics
            harmonic_info: Power in each harmonic
        """

        cancellation = torch.zeros_like(audio_signal)

        t = torch.arange(len(audio_signal), device=self.device) / self.sample_rate
        harmonic_info = {}

        for harmonic_idx in range(1, num_harmonics + 1):
            harmonic_freq = fundamental_freq_hz * harmonic_idx

            # Skip if beyond Nyquist
            if harmonic_freq > self.sample_rate / 2:
                continue

            # Estimate harmonic magnitude from spectrum
            spectrum = torch.abs(torch.fft.rfft(audio_signal, n=4096))
            freqs = torch.fft.rfftfreq(4096, 1.0 / self.sample_rate)

            # Find bin closest to harmonic
            bin_idx = torch.argmin(torch.abs(freqs - harmonic_freq))
            harmonic_power = float(spectrum[bin_idx])

            # Synthesize anti-phase
            harmonic_cancel = harmonic_power * torch.exp(
                1j * (2 * np.pi * harmonic_freq * t + np.pi)  # π phase = inversion
            )

            cancellation += harmonic_cancel
            harmonic_info[f'harmonic_{harmonic_idx}'] = {
                'freq_hz': float(harmonic_freq),
                'power_db': float(20 * torch.log10(torch.tensor(harmonic_power + 1e-9)))
            }

        return cancellation, harmonic_info


class SelectiveAudioSuppression:
    """
    Selectively cancel specific audio components while preserving others
    """

    def __init__(self, sample_rate_hz, device='cpu'):
        self.sample_rate = sample_rate_hz
        self.device = device

    def suppress_speech_band(self, audio_signal, speech_freq_min=80, speech_freq_max=8000,
                           suppression_db=20):
        """
        Attenuate speech frequencies while preserving others

        Useful for privacy or interference removal

        Args:
            audio_signal: Input audio
            speech_freq_min: Speech range minimum
            speech_freq_max: Speech range maximum
            suppression_db: Attenuation in dB

        Returns:
            suppressed_audio: Audio with speech band reduced
        """

        spectrum = torch.fft.fft(audio_signal)
        freqs = torch.fft.fftfreq(len(audio_signal), 1.0 / self.sample_rate)

        # Create bandpass suppression mask
        speech_mask = (torch.abs(freqs) >= speech_freq_min) & (torch.abs(freqs) <= speech_freq_max)

        # Attenuation factor
        atten_linear = 10 ** (-suppression_db / 20)

        # Apply suppression
        spectrum[speech_mask] *= atten_linear

        # IFFT
        suppressed = torch.fft.ifft(spectrum)

        return torch.real(suppressed)

    def suppress_background_noise(self, audio_signal, noise_profile, snr_threshold_db=10):
        """
        Spectral subtraction: remove noise based on learned profile

        Args:
            audio_signal: Input audio
            noise_profile: Noise power spectrum (learned from quiet period)
            snr_threshold_db: Minimum SNR to preserve signal

        Returns:
            denoised_audio: Background-reduced audio
        """

        spectrum = torch.abs(torch.fft.fft(audio_signal))

        # SNR-based mask
        snr_linear = (spectrum ** 2) / (noise_profile ** 2 + 1e-9)
        snr_db = 10 * torch.log10(snr_linear + 1e-9)

        # Binary mask: keep if SNR > threshold
        mask = (snr_db > snr_threshold_db).float()

        # Apply mask
        spectrum_masked = spectrum * mask

        # IFFT
        denoised = torch.fft.ifft(spectrum_masked)

        return torch.real(denoised)


if __name__ == "__main__":
    print("Active Audio Cancellation")
    print("=" * 70)

    sample_rate = 48000
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create test signal (audio-range tone + interference)
    duration = 0.5
    t = np.arange(int(duration * sample_rate)) / sample_rate

    target_tone = 0.5 * np.sin(2 * np.pi * 440 * t)  # A4 note
    interference = 0.3 * np.sin(2 * np.pi * 60 * t)  # 60 Hz mains

    audio = target_tone + interference
    audio = torch.tensor(audio, dtype=torch.float32, device=device)

    # Initialize canceller
    canceller = ActiveAudioCanceller(sample_rate, device=device)

    print("Estimating interference frequency...")
    freq, mag, conf = canceller.estimate_target_frequency(audio)
    print(f"  Detected frequency: {freq:.1f} Hz (confidence: {conf:.2f})")

    print("\nGenerating cancellation signal...")
    cancel_sig, info = canceller.synthesize_cancellation_signal(audio, target_freq=60)
    print(f"  Amplitude scale: {info['amplitude_scale']:.3f}")

    print("\nApplying cancellation...")
    residual, cancel_db, cancel_pct = canceller.predict_cancellation_error(audio, cancel_sig)
    print(f"  Cancellation: {cancel_db:.1f} dB ({cancel_pct:.1f}% reduction)")

    print("\nApplying notch filter...")
    filtered = canceller.frequency_domain_notch_filter(audio, center_freq_hz=60, bandwidth_hz=5)
    residual2, cancel_db2, _ = canceller.predict_cancellation_error(audio, filtered)
    print(f"  Notch filtering: {cancel_db2:.1f} dB")

    print("\nMulti-harmonic cancellation...")
    multi_cancel, harmonic_info = canceller.generate_harmonic_cancellation(
        audio, fundamental_freq_hz=60, num_harmonics=5
    )
    residual3, cancel_db3, _ = canceller.predict_cancellation_error(audio, multi_cancel)
    print(f"  Harmonic cancellation: {cancel_db3:.1f} dB")
