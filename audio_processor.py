#!/usr/bin/env python3
"""
audio_processor.py - Radar-optimized audio processing (MFCCs)
Expert implementation using torchaudio with SafeGEMM bypass for ROCm stability.
"""

import numpy as np
import torch
import torchaudio.transforms as T
from typing import Optional, Dict


class AudioProcessor:
    """
    Expert Acoustic Feature Extractor.
    Includes 'SafeGEMM' logic to bypass failing hipblas/matmul on some ROCm environments.
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
        
        # Test matmul for SafeGEMM decision
        self.use_safe_gemm = self._check_matmul_broken()
        
        # Standard transform (for CPU or working GPU)
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
        
        # Pre-allocate SafeGEMM buffers if needed
        if self.use_safe_gemm:
            self._setup_safe_gemm()

    def _check_matmul_broken(self):
        if self.device.type != 'cuda': return False
        try:
            torch.matmul(torch.randn(2, 2, device=self.device), 
                         torch.randn(2, 2, device=self.device))
            return False
        except:
            print("⚠️  AudioProcessor: Detected broken matmul, enabling SafeGEMM bypass.")
            return True

    def _setup_safe_gemm(self):
        """Prepare Mel filters and DCT matrix for manual application."""
        # Extract internal filters from torchaudio
        self.mel_fb = self.mfcc_transform.MelSpectrogram.mel_scale.fb.T.to(self.device) # (40, bins)
        # DCT Matrix (II)
        n_mels = 40
        n_mfcc = self.n_mfcc
        dct_mat = np.zeros((n_mfcc, n_mels))
        for k in range(n_mfcc):
            for n in range(n_mels):
                dct_mat[k, n] = np.cos(np.pi * k * (n + 0.5) / n_mels)
        self.dct_mat = torch.tensor(dct_mat, dtype=torch.float32, device=self.device)

    def _safe_matmul(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Apply matrix multiplication using sum() and broadcasting.
        Bypasses hipblas/matmul.
        A: (M, K), B: (K, N) -> Output: (M, N)
        """
        # Fully vectorized: (M, K, 1) * (1, K, N) -> (M, K, N)
        # Then sum over K dimension (dim=1)
        return torch.sum(A.unsqueeze(2) * B.unsqueeze(0), dim=1)

    def extract_features(self, signal: torch.Tensor) -> torch.Tensor:
        """Extract MFCCs using pure Torch or SafeGEMM bypass."""
        if not isinstance(signal, torch.Tensor):
            signal = torch.tensor(signal, dtype=torch.float32, device=self.device)
        
        if signal.is_complex():
            signal = torch.abs(signal).to(torch.float32)
        else:
            signal = signal.to(torch.float32)
            
        if signal.ndim == 1:
            signal = signal.unsqueeze(0)
            
        if not self.use_safe_gemm:
            return self.mfcc_transform(signal).squeeze(0)
        
        # --- SafeGEMM Pipeline ---
        # 1. Spectrogram
        spec = self.mfcc_transform.MelSpectrogram.spectrogram(signal).squeeze(0) # (bins, frames)
        
        # 2. Mel Scale (Manual Matmul)
        mel_spec = self._safe_matmul(self.mel_fb, spec)
        mel_spec_db = 20 * torch.log10(mel_spec + 1e-9)
        
        # 3. DCT (Manual Matmul)
        mfcc = self._safe_matmul(self.dct_mat, mel_spec_db)
        
        return mfcc

    def extract_voice_quality_metrics(self, signal: torch.Tensor) -> Dict[str, float]:
        """Estimate speech-likeness and quality."""
        if not isinstance(signal, torch.Tensor):
            signal = torch.tensor(signal, dtype=torch.float32, device=self.device)
        
        if signal.is_complex():
            signal = torch.abs(signal)

        spec = torch.abs(torch.fft.rfft(signal, n=self.n_fft))
        freqs = torch.fft.rfftfreq(self.n_fft, 1.0 / self.sample_rate).to(self.device)

        spec_centroid = torch.sum(freqs * spec) / (torch.sum(spec) + 1e-9)
        
        # Formant ratio logic
        formants = [700, 1220, 2600]
        formant_power = 0
        bin_size = self.sample_rate / self.n_fft
        for f in formants:
            idx = int(f / bin_size)
            if idx < len(spec): formant_power += spec[idx]

        total_power = torch.sum(spec)
        formant_ratio = formant_power / (total_power + 1e-9)

        return {
            'spectral_centroid_hz': float(spec_centroid.cpu().item()),
            'formant_ratio': float(formant_ratio.cpu().item()),
            'voice_likeness_score': float((formant_ratio * 10).cpu().item())
        }