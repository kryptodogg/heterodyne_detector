#!/usr/bin/env python3
"""
Radio-to-Acoustic Heterodyning Engine

Converts RF radar signals to audio-range frequencies via multi-stage heterodyning.
Produces audible "phantom voice" signatures for non-contact detection.

Theory: Multiple downconversions (RF → IF → AF) to translate GHz signals → audio (20-20kHz)
"""

import numpy as np
import torch
import scipy.signal as signal


class RadioToAcousticHeterodyner:
    """
    Multi-stage heterodyne downconverter: RF (GHz) → Audio (Hz)

    Example: 2.4 GHz radar signal
    Stage 1: Mix with 2.4 GHz LO → 100 kHz IF (intermediate frequency)
    Stage 2: Mix with 100 kHz LO → 1 kHz audio
    Stage 3: Mix with 1 kHz LO → 100 Hz phantom voice
    """

    def __init__(self, rf_freq_hz, target_audio_freq_hz, sample_rate_hz, device='cpu'):
        """
        Initialize heterodyner

        Args:
            rf_freq_hz: RF carrier frequency (e.g., 2.4e9 for 2.4 GHz)
            target_audio_freq_hz: Desired audio output frequency (50-4000 Hz typical)
            sample_rate_hz: ADC sample rate
            device: 'cpu' or 'cuda'
        """
        self.rf_freq = rf_freq_hz
        self.target_audio_freq = target_audio_freq_hz
        self.sample_rate = sample_rate_hz
        self.device = device

        # Design multi-stage conversion path
        self.stages = self._design_conversion_stages()

        # Pre-allocate reusable buffers
        self.max_buffer_size = int(sample_rate_hz * 1.0)  # 1 second max
        self.fft_buffer = torch.zeros(self.max_buffer_size,
                                      dtype=torch.complex64,
                                      device=device)

    def _design_conversion_stages(self):
        """
        Design frequency conversion path to reach audio range

        Returns list of (target_freq, lo_freq) tuples
        """
        stages = []

        # Stage 1: RF → Intermediate Frequency (IF)
        # Typical IF: 100 kHz - 1 MHz
        if_freq = 100e3  # 100 kHz IF

        stages.append({
            'name': 'RF to IF',
            'input_freq': self.rf_freq,
            'output_freq': if_freq,
            'lo_freq': self.rf_freq,  # Normally we'd track exact LO, use RF estimate
            'bandwidth': 100e3
        })

        # Stage 2: IF → Lower IF or Audio Baseband
        # Typical: mix IF down by another factor
        lower_if = self.target_audio_freq * 10  # 10x audio for intermediate stage
        if lower_if < 1e3:
            lower_if = 1e3  # Minimum 1 kHz

        stages.append({
            'name': 'IF to Lower IF',
            'input_freq': if_freq,
            'output_freq': lower_if,
            'lo_freq': if_freq,
            'bandwidth': lower_if
        })

        # Stage 3: Lower IF → Audio Baseband
        stages.append({
            'name': 'Lower IF to Audio',
            'input_freq': lower_if,
            'output_freq': self.target_audio_freq,
            'lo_freq': lower_if,
            'bandwidth': self.target_audio_freq
        })

        return stages

    def heterodyne_iq(self, iq_signal, stage_index=0):
        """
        Single-stage heterodyne mixing

        Args:
            iq_signal: Complex IQ signal (RF or IF)
            stage_index: Which stage (0=RF→IF, 1=IF→LowerIF, etc.)

        Returns:
            Mixed and filtered output (next stage frequency)
        """
        if stage_index >= len(self.stages):
            raise ValueError(f"Stage {stage_index} out of range")

        stage = self.stages[stage_index]
        lo_freq = stage['lo_freq']

        # Time vector
        t = torch.arange(len(iq_signal), device=self.device) / self.sample_rate

        # Local oscillator (complex exponential)
        lo = torch.exp(-2j * np.pi * lo_freq * t)

        # Mix: s(t) * LO(t)
        mixed = iq_signal * lo

        # Low-pass filter to remove sum frequency
        # Design Butterworth LPF at output bandwidth
        output_bw = stage['output_freq']
        nyquist = self.sample_rate / 2

        if output_bw > nyquist * 0.4:
            # Can't filter at this rate, just decimate
            decimation_factor = max(1, int(self.sample_rate / (4 * output_bw)))
        else:
            decimation_factor = 1

        # Simple moving average low-pass (for GPU efficiency)
        if decimation_factor > 1:
            kernel_size = decimation_factor
            # Convert to numpy for filtering, convert back
            mixed_np = mixed.cpu().numpy() if isinstance(mixed, torch.Tensor) else mixed
            filtered = signal.decimate(mixed_np, decimation_factor, zero_phase=True)
            filtered = torch.tensor(filtered, dtype=torch.complex64, device=self.device)
        else:
            filtered = mixed

        return filtered

    def full_heterodyne_pipeline(self, rf_iq_signal):
        """
        Complete RF → Audio conversion pipeline

        Args:
            rf_iq_signal: RF IQ signal (complex, pre-demodulated)

        Returns:
            audio_signal: Audio-range complex signal
        """

        signal_stage = rf_iq_signal.clone()

        for i, stage in enumerate(self.stages):
            signal_stage = self.heterodyne_iq(signal_stage, stage_index=i)

        # Extract magnitude (audio envelope)
        audio_envelope = torch.abs(signal_stage)

        # Extract phase (pitch information via instantaneous frequency)
        audio_phase = torch.angle(signal_stage)

        return {
            'complex': signal_stage,
            'envelope': audio_envelope,
            'phase': audio_phase,
            'magnitude_db': 20 * torch.log10(audio_envelope + 1e-12)
        }

    def synthesize_audio_waveform(self, audio_complex):
        """
        Convert complex audio-range signal to real waveform

        Args:
            audio_complex: Complex signal at audio frequencies

        Returns:
            audio_waveform: Real-valued audio (can be played)
        """

        # Method 1: Use envelope + instantaneous phase
        # This preserves amplitude modulation (voice formants)
        audio_waveform = torch.real(audio_complex)

        # Normalize to [-1, 1] for audio playback
        max_val = torch.max(torch.abs(audio_waveform))
        if max_val > 0:
            audio_waveform = audio_waveform / max_val

        return audio_waveform

    def extract_formant_frequencies(self, audio_complex, fft_size=4096):
        """
        Extract formant frequencies from audio signal
        (speech-like characteristics of heterodyned radar)

        Args:
            audio_complex: Complex audio signal
            fft_size: FFT size for spectral analysis

        Returns:
            formant_freqs: List of dominant frequencies (formants)
            formant_powers: Power at each formant
        """

        # Take magnitude spectrum
        spectrum = torch.abs(torch.fft.fft(audio_complex, n=fft_size))

        # Frequency bins
        freqs = torch.fft.fftfreq(fft_size, 1.0 / self.sample_rate)

        # Audio range only (20 Hz - 4 kHz)
        audio_mask = (freqs > 20) & (freqs < 4000)
        spectrum_audio = spectrum[audio_mask]
        freqs_audio = freqs[audio_mask]

        # Find peaks (formants)
        peaks, properties = signal.find_peaks(
            spectrum_audio.cpu().numpy(),
            height=torch.max(spectrum_audio).item() * 0.1,  # 10% of max
            distance=self.sample_rate / fft_size / 100  # Min 100 Hz apart
        )

        formant_freqs = freqs_audio[peaks]
        formant_powers = spectrum_audio[peaks]

        # Sort by power
        sorted_idx = torch.argsort(formant_powers, descending=True)

        return formant_freqs[sorted_idx][:5], formant_powers[sorted_idx][:5]  # Top 5


class AcousticCarrierReconstructor:
    """
    Generate reference carriers for heterodyne detection at audio frequencies

    Used to phase-lock audio output or detect specific acoustic patterns
    """

    def __init__(self, target_audio_freq_hz, sample_rate_hz, device='cpu'):
        self.target_freq = target_audio_freq_hz
        self.sample_rate = sample_rate_hz
        self.device = device

    def generate_reference_carrier(self, duration_sec, phase_offset_deg=0):
        """
        Generate clean reference carrier at target audio frequency

        Args:
            duration_sec: Duration in seconds
            phase_offset_deg: Initial phase offset

        Returns:
            carrier: Complex carrier signal
        """

        n_samples = int(duration_sec * self.sample_rate)
        t = torch.arange(n_samples, device=self.device) / self.sample_rate

        phase_offset_rad = np.radians(phase_offset_deg)

        carrier = torch.exp(
            2j * np.pi * self.target_freq * t + 1j * phase_offset_rad
        )

        return carrier

    def phase_lock_detector(self, audio_signal, lock_bandwidth_hz=10):
        """
        Phase-locked loop to track audio signal frequency

        Useful for extracting Doppler shifts or modulation patterns

        Args:
            audio_signal: Complex audio signal
            lock_bandwidth_hz: Loop bandwidth

        Returns:
            instantaneous_freq: Estimated frequency at each sample
            phase_error: Phase deviation from target
        """

        # Unwrap phase
        phase = torch.angle(audio_signal)
        phase_unwrapped = torch.cumsum(
            torch.where(
                torch.abs(torch.diff(phase, prepend=phase[0:1])) > np.pi,
                torch.diff(phase, prepend=phase[0:1]) - 2 * np.pi,
                torch.diff(phase, prepend=phase[0:1])
            ),
            dim=0
        )

        # Instantaneous frequency = rate of phase change
        inst_freq = torch.diff(phase_unwrapped, prepend=phase_unwrapped[0:1]) * self.sample_rate / (2 * np.pi)

        # Low-pass filter to get smooth frequency estimate
        kernel_size = max(1, int(self.sample_rate / lock_bandwidth_hz / 100))
        if kernel_size > 1:
            inst_freq_smooth = torch.nn.functional.avg_pool1d(
                inst_freq.unsqueeze(0).unsqueeze(0),
                kernel_size=kernel_size,
                padding=kernel_size//2
            ).squeeze()
        else:
            inst_freq_smooth = inst_freq

        # Phase error from target frequency
        phase_error = inst_freq_smooth - self.target_freq

        return inst_freq_smooth, phase_error


class PhantomVoiceSynthesizer:
    """
    Generate "phantom voice" patterns from heterodyned radar

    The heterodyning process creates a unique modulation pattern that,
    when rendered as audio, sounds speech-like or exhibits characteristic
    prosody (pitch variation, formants).

    This synthesizer enhances and preserves these patterns.
    """

    def __init__(self, sample_rate_hz, device='cpu'):
        self.sample_rate = sample_rate_hz
        self.device = device

    def apply_vocal_tract_filter(self, audio_signal, formant_freqs=None):
        """
        Apply vocal tract-like filtering to enhance speech characteristics

        Args:
            audio_signal: Audio signal
            formant_freqs: Formant frequencies [f1, f2, f3] (default: human-like)

        Returns:
            filtered_signal: Audio with vocal tract coloration
        """

        if formant_freqs is None:
            # Default human formants (male speaker)
            formant_freqs = [700, 1220, 2600]  # Hz

        # Design cascade of resonant filters
        filtered = audio_signal.clone()

        for f_formant in formant_freqs:
            # Second-order resonant filter (peaking EQ)
            Q = 5.0  # Resonance width
            w0 = 2 * np.pi * f_formant / self.sample_rate

            # Transfer function
            alpha = np.sin(w0) / (2 * Q)

            b0 = 1 + alpha
            b1 = -2 * np.cos(w0)
            b2 = 1 - alpha
            a0 = 1 + alpha
            a1 = -2 * np.cos(w0)
            a2 = 1 - alpha

            # Normalize
            b = torch.tensor([b0, b1, b2], dtype=torch.float32, device=self.device) / a0
            a = torch.tensor([1.0, a1/a0, a2/a0], dtype=torch.float32, device=self.device)

            # Apply filter
            filtered_np = signal.lfilter(b.cpu().numpy(), a.cpu().numpy(),
                                        filtered.cpu().numpy())
            filtered = torch.tensor(filtered_np, dtype=audio_signal.dtype, device=self.device)

        return filtered

    def add_prosody_modulation(self, audio_signal, pitch_contour=None, vibrato_rate_hz=5):
        """
        Add pitch modulation (prosody) to simulate speech patterns

        Args:
            audio_signal: Audio signal
            pitch_contour: Optional pitch envelope (normalized 0-1)
            vibrato_rate_hz: Vibrato frequency (typical 4-8 Hz)

        Returns:
            modulated_signal: Audio with pitch variations
        """

        t = torch.arange(len(audio_signal), device=self.device) / self.sample_rate

        if pitch_contour is None:
            # Generate natural pitch contour (rises then falls)
            pitch_contour = torch.sin(2 * np.pi * 0.5 * t)  # 0.5 Hz oscillation
            pitch_contour = (pitch_contour + 1) / 2  # Normalize to [0, 1]

        # Vibrato modulation
        vibrato = 0.1 * torch.sin(2 * np.pi * vibrato_rate_hz * t)

        # Phase modulation
        phase_modulation = pitch_contour * 100 + vibrato  # ±100 Hz pitch variation

        # Apply phase modulation via complex multiplication
        phase_shift = torch.exp(1j * 2 * np.pi * phase_modulation * t)
        modulated = audio_signal * phase_shift

        return torch.real(modulated)

    def extract_and_enhance_speech_formants(self, audio_signal, num_formants=3):
        """
        Identify dominant formants and enhance them

        Creates more speech-like perception

        Args:
            audio_signal: Audio signal
            num_formants: Number of formants to extract

        Returns:
            enhanced_signal: Audio with enhanced formants
        """

        # Compute power spectrum
        fft_size = min(4096, len(audio_signal))
        spectrum = torch.abs(torch.fft.fft(audio_signal, n=fft_size))
        freqs = torch.fft.fftfreq(fft_size, 1.0 / self.sample_rate)

        # Find peaks (formants)
        spec_np = spectrum.cpu().numpy()
        peaks, _ = signal.find_peaks(spec_np, height=np.max(spec_np) * 0.1)

        # Get top formants
        top_peaks = peaks[np.argsort(spec_np[peaks])[-num_formants:]]
        formant_freqs = freqs[top_peaks].cpu().numpy()

        # Apply vocal tract filter with extracted formants
        enhanced = self.apply_vocal_tract_filter(audio_signal, formant_freqs=formant_freqs)

        return enhanced


if __name__ == "__main__":
    print("Radio-to-Acoustic Heterodyning Engine")
    print("=" * 70)

    # Parameters
    rf_freq = 2.4e9  # 2.4 GHz
    target_audio_freq = 500  # 500 Hz target
    sample_rate = 10e6  # 10 MHz ADC
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"RF Frequency: {rf_freq/1e9:.1f} GHz")
    print(f"Target Audio: {target_audio_freq:.0f} Hz")
    print(f"Sample Rate: {sample_rate/1e6:.1f} MHz")
    print(f"Device: {device}\n")

    # Initialize heterodyner
    het = RadioToAcousticHeterodyner(rf_freq, target_audio_freq, sample_rate, device=device)

    # Create test RF signal (simulated 2.4 GHz)
    duration = 0.1  # 100 ms
    t = np.arange(int(duration * sample_rate)) / sample_rate

    # RF IQ signal with modulation
    rf_iq = torch.tensor(
        np.exp(1j * 2 * np.pi * 100e3 * t) * (1 + 0.5 * np.sin(2 * np.pi * 50 * t)),
        dtype=torch.complex64,
        device=device
    )

    print("Converting RF signal to audio...")
    audio = het.full_heterodyne_pipeline(rf_iq)

    print(f"Audio envelope shape: {audio['envelope'].shape}")
    print(f"Audio magnitude range: {audio['magnitude_db'].min():.1f} to {audio['magnitude_db'].max():.1f} dB")

    # Extract formants
    formants, powers = het.extract_formant_frequencies(audio['complex'])
    print(f"\nExtracted formants: {formants[:3].cpu().numpy()}")

    # Synthesize audio waveform
    audio_wave = het.synthesize_audio_waveform(audio['complex'])
    print(f"Audio waveform: {audio_wave.shape}, range: [{audio_wave.min():.3f}, {audio_wave.max():.3f}]")

    # Phantom voice synthesis
    synth = PhantomVoiceSynthesizer(sample_rate, device=device)
    enhanced = synth.extract_and_enhance_speech_formants(audio['complex'])
    print(f"Enhanced audio generated: {enhanced.shape}")
