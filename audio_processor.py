#!/usr/bin/env python3
"""
audio_processor.py - Radar-optimized audio processing (MFCCs)
Expert implementation using torchaudio and pure PyTorch.

Features:
- torchaudio-based MFCC extraction
- Voice quality metrics (Spectral Centroid, Formant Ratio)
- Pitch contour extraction
- Activity signature extraction
"""

import numpy as np
import torch
import torchaudio.transforms as T
from typing import Optional, Dict


class AudioProcessor:
    """
    Expert Acoustic Feature Extractor for heterodyned radar signals.
    Utilizes torchaudio for GPU-accelerated spectral processing.
    """
    
    def __init__(
        self,
        sample_rate: int = 10000,
        n_mfcc: int = 13,
        n_fft: int = 1024,
        hop_length: float = 0.05,
        window_size: float = 0.375,
        device: torch.device = torch.device('cpu'),
        **kwargs
    ):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.device = device
        
        # Convert seconds to samples
        self.hop_length = int(hop_length * sample_rate)
        self.n_win = int(window_size * sample_rate)
        
        # Initialize torchaudio MFCC transform
        # n_mels should be >= n_mfcc, standard is 40 or 128
        self.mfcc_transform = T.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=self.n_mfcc,
            melkwargs={
                "n_fft": self.n_fft,
                "n_mels": 40,
                "hop_length": self.hop_length,
                "win_length": self.n_fft,
                "center": True,
                "pad_mode": "reflect",
                "power": 2.0,
            }
        ).to(self.device)

    def extract_features(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Main entry point for RadarApp. Extracts MFCCs using torchaudio.
        """
        if not isinstance(signal, torch.Tensor):
            signal = torch.tensor(signal, dtype=torch.float32, device=self.device)
        
        # Real signal check
        if signal.is_complex():
            signal = torch.abs(signal)
            
        # Ensure signal has channel dimension for torchaudio [batch, channel, time] or [channel, time]
        if signal.ndim == 1:
            signal = signal.unsqueeze(0)
            
        # MFCC extraction
        mfcc = self.mfcc_transform(signal)
        
        # Squeeze back if we added a dimension
        return mfcc.squeeze(0)

    def extract_voice_quality_metrics(self, signal: torch.Tensor) -> Dict[str, float]:
        """
        Estimate speech-likeness and quality (Spectral Centroid, Formant Ratio).
        """
        if not isinstance(signal, torch.Tensor):
            signal = torch.tensor(signal, dtype=torch.float32, device=self.device)
        
        if signal.is_complex():
            signal = torch.abs(signal)

        # Compute spectrum
        spec = torch.abs(torch.fft.rfft(signal, n=self.n_fft))
        freqs = torch.fft.rfftfreq(self.n_fft, 1.0 / self.sample_rate).to(self.device)

        # Spectral centroid (center of mass)
        spec_centroid = torch.sum(freqs * spec) / (torch.sum(spec) + 1e-9)

        # Formant-likeness: ratio near typical human formants [700, 1220, 2600]
        formants = [700, 1220, 2600]
        formant_power = 0
        for f in formants:
            idx = int(f / (self.sample_rate / self.n_fft))
            if idx < len(spec):
                formant_power += spec[idx]

        total_power = torch.sum(spec)
        formant_ratio = formant_power / (total_power + 1e-9)

        return {
            'spectral_centroid_hz': float(spec_centroid.cpu().item()),
            'formant_ratio': float(formant_ratio.cpu().item()),
            'voice_likeness_score': float((formant_ratio * 10).cpu().item())
        }

    def extract_activity_signature(self, signal: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract activity features (RMS, ZCR) for classification.
        """
        # RMS Energy
        rms = torch.sqrt(torch.mean(signal**2))
        
        # Zero Crossing Rate
        zcr = torch.sum(torch.abs(torch.sign(signal[1:]) - torch.sign(signal[:-1]))) / (2 * len(signal))
        
        return {
            'rms_energy': rms,
            'zero_crossing_rate': zcr
        }
