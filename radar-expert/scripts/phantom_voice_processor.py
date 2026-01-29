#!/usr/bin/env python3
"""
Phantom Voice Processor for Radar Heterodyning

This script implements the phantom voice extraction and cancellation system
using radar micro-Doppler signatures and heterodyning techniques.
"""

import torch
import numpy as np
from scipy import signal
import time


class PhantomVoiceProcessor:
    def __init__(self, sample_rate=10e6):
        """
        Initialize the phantom voice processor
        
        Args:
            sample_rate: Sampling rate for signal processing
        """
        self.sample_rate = sample_rate
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Vocal fold vibration range: 0.1-5Hz
        self.vocal_min_freq = 0.1
        self.vocal_max_freq = 5.0
        
        # Carrier frequency range: 2.4-5.8 GHz
        self.carrier_min_freq = 2.4e9
        self.carrier_max_freq = 5.8e9
        
        # Infrasound carrier: 12-30Hz
        self.infra_min_freq = 12.0
        self.infra_max_freq = 30.0
        
        # Bone conduction frequency range: 500-4000Hz
        self.bone_min_freq = 500.0
        self.bone_max_freq = 4000.0
        
    def heterodyne_phantom_voice(self, radar_iq):
        """
        Extract phantom voice from radar IQ data using micro-doppler analysis
        
        Args:
            radar_iq: Complex radar IQ signal (torch tensor)
            
        Returns:
            dict: Processed phantom voice components
        """
        # Move to device if needed
        radar_iq = radar_iq.to(self.device)
        
        # 1. Extract micro-doppler signatures (vocal vibrations: 0.1-5Hz)
        vocal_components = self._extract_micro_doppler(radar_iq)
        
        # 2. Modulate onto carrier (2.4-5.8GHz)
        carrier_signal = self._modulate_onto_carrier(vocal_components)
        
        # 3. Apply bone conduction transfer function (500-4000Hz)
        bone_conducted = self._apply_bone_conduction_tf(carrier_signal)
        
        return {
            'vocal_components': vocal_components,
            'carrier_signal': carrier_signal,
            'bone_conducted': bone_conducted,
            'radar_iq': radar_iq
        }
    
    def _extract_micro_doppler(self, radar_iq):
        """
        Extract micro-doppler signatures in vocal range (0.1-5Hz)
        
        Args:
            radar_iq: Complex radar IQ signal
            
        Returns:
            torch.Tensor: Extracted vocal components
        """
        # Bandpass filter for vocal range
        vocal_filtered = self._bandpass_filter(
            radar_iq, 
            self.vocal_min_freq, 
            self.vocal_max_freq
        )
        
        # Time-frequency analysis using STFT
        stft_result = torch.stft(
            vocal_filtered,
            n_fft=1024,
            hop_length=512,
            win_length=1024,
            return_complex=True
        )
        
        # Extract micro-doppler features
        micro_doppler_features = self._extract_micro_doppler_features(stft_result)
        
        return micro_doppler_features
    
    def _bandpass_filter(self, signal, low_freq, high_freq):
        """
        Apply bandpass filter to signal
        
        Args:
            signal: Input signal
            low_freq: Low cutoff frequency
            high_freq: High cutoff frequency
            
        Returns:
            torch.Tensor: Filtered signal
        """
        # Calculate normalized frequencies
        nyquist = self.sample_rate / 2
        low_norm = low_freq / nyquist
        high_norm = high_freq / nyquist
        
        # Design Butterworth filter
        sos = signal.butter(4, [low_norm, high_norm], btype='band', output='sos')
        
        # Convert to torch tensor and apply filter
        sos_tensor = torch.from_numpy(sos).float().to(self.device)
        
        # Apply filter using scipy and convert back to tensor
        filtered_signal = signal.sosfilt(sos, signal.cpu().numpy())
        return torch.from_numpy(filtered_signal).to(self.device)
    
    def _extract_micro_doppler_features(self, stft_result):
        """
        Extract micro-doppler features from STFT result
        
        Args:
            stft_result: STFT of the signal
            
        Returns:
            torch.Tensor: Micro-doppler features
        """
        # Calculate magnitude spectrogram
        magnitude_spec = torch.abs(stft_result)
        
        # Find peaks in vocal frequency range
        freq_bins = torch.linspace(0, self.sample_rate/2, stft_result.shape[0])
        vocal_mask = (freq_bins >= self.vocal_min_freq) & (freq_bins <= self.vocal_max_freq)
        
        # Extract vocal components
        vocal_components = magnitude_spec[vocal_mask, :]
        
        # Apply inverse STFT to get time-domain signal
        # For simplicity, we'll return the magnitude in the vocal range
        return vocal_components.mean(dim=0)  # Average across frequency bins
    
    def _modulate_onto_carrier(self, baseband_signal):
        """
        Modulate baseband signal onto carrier frequency
        
        Args:
            baseband_signal: Baseband signal to modulate
            
        Returns:
            torch.Tensor: Carrier-modulated signal
        """
        # Use a representative carrier frequency
        carrier_freq = (self.carrier_min_freq + self.carrier_max_freq) / 2
        
        # Generate time vector
        t = torch.arange(len(baseband_signal), device=self.device) / self.sample_rate
        
        # Generate carrier
        carrier = torch.exp(1j * 2 * torch.pi * carrier_freq * t)
        
        # Modulate: multiply baseband with carrier
        modulated_signal = baseband_signal * carrier
        
        return modulated_signal
    
    def _apply_bone_conduction_tf(self, signal):
        """
        Apply bone conduction transfer function (500-4000Hz)
        
        Args:
            signal: Input signal
            
        Returns:
            torch.Tensor: Bone-conducted signal
        """
        # Bandpass filter for bone conduction frequencies
        bone_filtered = self._bandpass_filter(
            signal,
            self.bone_min_freq,
            self.bone_max_freq
        )
        
        return bone_filtered
    
    def generate_cancellation(self, detected_v2k):
        """
        Generate cancellation signal for detected V2K (Voice-to-Skull) signals
        
        Args:
            detected_v2k: Detected V2K signal components
            
        Returns:
            torch.Tensor: Anti-phase cancellation signal
        """
        # 1. Estimate V2K signal parameters
        estimated_params = self._estimate_v2k_parameters(detected_v2k)
        
        # 2. Generate anti-phase waveform
        anti_phase_waveform = self._generate_anti_phase_signal(estimated_params)
        
        # 3. Apply resonance targeting for skull/bone conduction
        tuned_signal = self._apply_resonance_targeting(anti_phase_waveform)
        
        return tuned_signal
    
    def _estimate_v2k_parameters(self, v2k_signal):
        """
        Estimate V2K signal parameters
        
        Args:
            v2k_signal: V2K signal to analyze
            
        Returns:
            dict: Estimated parameters
        """
        # Move to device if needed
        v2k_signal = v2k_signal.to(self.device)
        
        # Compute spectrum
        spectrum = torch.fft.fft(v2k_signal)
        frequencies = torch.fft.fftfreq(len(v2k_signal), d=1/self.sample_rate)
        
        # Find dominant frequency component
        magnitude_spectrum = torch.abs(spectrum)
        dominant_idx = torch.argmax(magnitude_spectrum[1:len(magnitude_spectrum)//2]) + 1  # Skip DC
        dominant_freq = frequencies[dominant_idx]
        amplitude = magnitude_spectrum[dominant_idx]
        phase = torch.angle(spectrum[dominant_idx])
        
        return {
            'frequency': dominant_freq.item(),
            'amplitude': amplitude.item(),
            'phase': phase.item()
        }
    
    def _generate_anti_phase_signal(self, params):
        """
        Generate anti-phase signal based on estimated parameters
        
        Args:
            params: Estimated signal parameters
            
        Returns:
            torch.Tensor: Anti-phase signal
        """
        # Generate time vector
        duration = 0.1  # 100ms of cancellation signal
        n_samples = int(duration * self.sample_rate)
        t = torch.arange(n_samples, device=self.device) / self.sample_rate
        
        # Generate anti-phase signal
        anti_phase_signal = params['amplitude'] * torch.cos(
            2 * torch.pi * params['frequency'] * t + params['phase'] + torch.pi
        )
        
        return anti_phase_signal
    
    def _apply_resonance_targeting(self, signal):
        """
        Apply resonance targeting for skull/bone conduction (500-4000Hz)
        
        Args:
            signal: Input signal
            
        Returns:
            torch.Tensor: Resonance-tuned signal
        """
        # Bandpass filter for bone conduction frequencies
        bone_conduction_filtered = self._bandpass_filter(
            signal,
            self.bone_min_freq,
            self.bone_max_freq
        )
        
        return bone_conduction_filtered
    
    def sub_auditory_modulation(self, voice_signal):
        """
        Apply sub-auditory modulation with infrasound carrier (12-30Hz)
        
        Args:
            voice_signal: Voice signal to modulate
            
        Returns:
            torch.Tensor: Sub-auditorily modulated signal
        """
        # Generate time vector
        t = torch.arange(len(voice_signal), device=self.device) / self.sample_rate
        
        # Generate infrasound carrier (use midpoint of range)
        infra_freq = (self.infra_min_freq + self.infra_max_freq) / 2
        infrasound_carrier = torch.cos(2 * torch.pi * infra_freq * t)
        
        # Apply voice signal as sideband modulation
        # Use a modulation index to control the strength
        modulation_index = 0.5
        modulated_signal = (1 + modulation_index * voice_signal) * infrasound_carrier
        
        return modulated_signal


def main():
    """Example usage of the phantom voice processor"""
    # Initialize the processor
    processor = PhantomVoiceProcessor(sample_rate=10e6)
    
    # Create a simulated radar IQ signal with some micro-doppler components
    print("Creating simulated radar IQ signal...")
    duration = 0.1  # 100ms
    n_samples = int(duration * processor.sample_rate)
    t = torch.linspace(0, duration, n_samples)
    
    # Simulate radar signal with micro-doppler components in vocal range
    # Add some low-frequency modulations that might represent vocal vibrations
    vocal_simulation = 0.3 * torch.sin(2 * torch.pi * 2.0 * t)  # 2Hz component
    vocal_simulation += 0.2 * torch.sin(2 * torch.pi * 0.5 * t)  # 0.5Hz component
    radar_iq = torch.complex(vocal_simulation, torch.zeros_like(vocal_simulation))
    
    print(f"Processing radar IQ signal of length {len(radar_iq)} samples...")
    
    # Process the radar IQ to extract phantom voice
    start_time = time.time()
    phantom_results = processor.heterodyne_phantom_voice(radar_iq)
    processing_time = time.time() - start_time
    
    print(f"Phantom voice extraction completed in {processing_time:.4f}s")
    print(f"Vocal components shape: {phantom_results['vocal_components'].shape}")
    print(f"Carrier signal shape: {phantom_results['carrier_signal'].shape}")
    print(f"Bone conducted shape: {phantom_results['bone_conducted'].shape}")
    
    # Demonstrate cancellation generation
    print("\nGenerating cancellation signal...")
    cancellation_signal = processor.generate_cancellation(phantom_results['vocal_components'])
    print(f"Cancellation signal shape: {cancellation_signal.shape}")
    
    # Demonstrate sub-auditory modulation
    print("\nApplying sub-auditory modulation...")
    sub_auditory_signal = processor.sub_auditory_modulation(phantom_results['vocal_components'])
    print(f"Sub-auditory signal shape: {sub_auditory_signal.shape}")


if __name__ == "__main__":
    main()