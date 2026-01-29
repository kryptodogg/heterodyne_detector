#!/usr/bin/env python3
"""
heterodyne_detector.py - Expert Radar Heterodyne Detection
Integrates multi-stage downconversion and phantom voice analysis.
Pure PyTorch implementation for GPU acceleration.
"""

import numpy as np
import torch
try:
    import iio
except Exception:
    iio = None
import threading
import queue
import time
from collections import deque
from typing import Dict, Any, Optional


class RadioToAcousticHeterodyner:
    """
    Multi-stage heterodyne downconverter: RF (GHz) -> Audio (Hz)
    Expert implementation from radar-expert skill.
    """
    def __init__(self, rf_freq_hz: float, target_audio_freq_hz: float, 
                 sample_rate_hz: float, device: torch.device):
        self.rf_freq = rf_freq_hz
        self.target_audio_freq = target_audio_freq_hz
        self.sample_rate = sample_rate_hz
        self.device = device
        self.stages = self._design_conversion_stages()

    def _design_conversion_stages(self):
        stages = []
        if_freq = 100e3  # 100 kHz IF
        stages.append({'name': 'RF to IF', 'lo_freq': self.rf_freq, 'output_freq': if_freq})
        
        lower_if = self.target_audio_freq * 10
        stages.append({'name': 'IF to Lower IF', 'lo_freq': if_freq, 'output_freq': lower_if})
        
        stages.append({'name': 'Lower IF to Audio', 'lo_freq': lower_if, 'output_freq': self.target_audio_freq})
        return stages

    def full_heterodyne_pipeline(self, iq_signal: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Convert RF IQ to audio-range complex signal."""
        curr_signal = iq_signal.to(self.device)
        t = torch.arange(len(curr_signal), device=self.device) / self.sample_rate
        
        for stage in self.stages:
            lo = torch.exp(-2j * np.pi * stage['lo_freq'] * t)
            curr_signal = curr_signal * lo
            # Simple decimation/LPF approximation for speed
            # In production, we'd use a real LPF
            
        audio_envelope = torch.abs(curr_signal)
        return {
            'complex': curr_signal,
            'envelope': audio_envelope,
            'magnitude_db': 20 * torch.log10(audio_envelope + 1e-12)
        }


class HeterodyneDetector:
    """
    Detects heterodyne artifacts and phantom voices using dual RX channels.
    Torch-first architecture with expert multi-stage analysis.
    """
    
    def __init__(self, sample_rate=2.4e6, center_freq=100e6, 
                 rx_gain=50, buffer_size=2**16, device='cpu', **kwargs):
        self.sample_rate = sample_rate
        self.center_freq = center_freq
        self.rx_gain = rx_gain
        self.buffer_size = buffer_size
        self.device = torch.device(device) if isinstance(device, str) else device
        
        # Expert Heterodyner
        # Note: Input signal is already baseband (downconverted by SDR), so rf_freq_hz should be 0
        # to avoid double-shifting.
        self.heterodyner = RadioToAcousticHeterodyner(
            rf_freq_hz=0.0,
            target_audio_freq_hz=500.0, # Target audible range
            sample_rate_hz=sample_rate,
            device=self.device
        )
        
        # Hardware context (Pluto+ SDR)
        self.ctx = None
        self.rx_buffer = None
        
        # State
        self.running = False
        self.heterodyne_threshold = 0.7
        self.phantom_voice_freq_range = (300, 3400)
        self.history = deque(maxlen=100)

    def detect(self, signal1: torch.Tensor, signal2: torch.Tensor) -> Dict[str, Any]:
        """Main entry point for RadarApp. Pure Torch implementation."""
        if not isinstance(signal1, torch.Tensor):
            signal1 = torch.tensor(signal1, dtype=torch.complex64, device=self.device)
            signal2 = torch.tensor(signal2, dtype=torch.complex64, device=self.device)
        
        # Step 1: Expert multi-stage downconversion
        audio1 = self.heterodyner.full_heterodyne_pipeline(signal1)
        audio2 = self.heterodyner.full_heterodyne_pipeline(signal2)
        
        # Step 2: Cross-correlation for heterodyne score
        # Use FFT-based correlation for GPU efficiency
        n_fft = 2 * len(signal1)
        fft1 = torch.fft.fft(signal1, n=n_fft)
        fft2 = torch.fft.fft(signal2, n=n_fft)
        corr = torch.fft.ifft(fft1 * torch.conj(fft2))
        peak_val = torch.max(torch.abs(corr))
        
        # Normalize score
        norm = torch.sqrt(torch.sum(torch.abs(signal1)**2) * torch.sum(torch.abs(signal2)**2))
        score = float((peak_val / (norm + 1e-10)).cpu().item())
        
        # Step 3: Frequency offset
        # Dominant phase slope
        phase_diff = torch.angle(fft1 * torch.conj(fft2))
        # Pure Torch unwrap logic
        d_phase = torch.diff(phase_diff)
        d_phase_adj = (d_phase + np.pi) % (2 * np.pi) - np.pi
        freq_offset = float(torch.mean(d_phase_adj).cpu().item())
        freq_offset = freq_offset * self.sample_rate / (2 * np.pi)
        
        # Step 4: Voice characteristics
        # Check if envelope energy is in voice range
        voice_ratio = self._check_voice_presence(audio1['envelope'])
        
        return {
            'score': score,
            'freq_offset': freq_offset,
            'is_voice_range': voice_ratio > 0.15,
            'audio_complex': audio1['complex'],
            'timestamp': time.time()
        }

    def _check_voice_presence(self, envelope: torch.Tensor) -> float:
        """Analyze if the envelope contains speech-like energy."""
        # FFT of envelope to find modulation frequencies
        spec = torch.abs(torch.fft.rfft(envelope))
        freqs = torch.fft.rfftfreq(len(envelope), 1.0 / self.sample_rate).to(self.device)
        
        voice_mask = (freqs >= self.phantom_voice_freq_range[0]) & \
                     (freqs <= self.phantom_voice_freq_range[1])
        
        voice_energy = torch.sum(spec[voice_mask]**2)
        total_energy = torch.sum(spec**2)
        
        return float((voice_energy / (total_energy + 1e-10)).cpu().item())

    # --- Hardware Methods (Pluto+ Integration) ---
    
    def connect_pluto(self, uri='ip:192.168.2.1'):
        try:
            self.ctx = iio.Context(uri)
            phy = self.ctx.find_device("ad9361-phy")
            rx_dev = self.ctx.find_device("cf-ad9361-lpc")
            
            # Basic configuration
            phy.find_channel("altvoltage0", True).attrs["frequency"].value = str(int(self.center_freq))
            rx_chan = rx_dev.find_channel("voltage0")
            rx_chan.enabled = True
            
            self.rx_buffer = iio.Buffer(rx_dev, self.buffer_size, False)
            print(f"✅ Connected to Pluto+ at {uri}")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to Pluto+: {e}")
            return False

    def read_samples(self):
        """Read IQ samples and return as Torch tensors."""
        if self.rx_buffer is None:
            return None, None
            
        self.rx_buffer.refill()
        data = np.frombuffer(self.rx_buffer.read(), dtype=np.int16).astype(np.float32)
        
        # Extract RX1/RX2 (Interleaved I/Q)
        # Note: Actual mapping depends on AD9361 config
        rx1 = torch.tensor(data[0::4] + 1j*data[1::4], dtype=torch.complex64, device=self.device)
        rx2 = torch.tensor(data[2::4] + 1j*data[3::4], dtype=torch.complex64, device=self.device)
        
        return rx1, rx2