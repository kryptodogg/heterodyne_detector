#!/usr/bin/env python3
"""Radar-optimized audio processing (MFCCs) - minimal synchronous implementation."""

import numpy as np
import librosa
import torch
from typing import Optional


class AudioProcessor:
    def __init__(
        self,
        sample_rate: int = 2400000,
        n_mfcc: int = 13,
        n_fft: int = 1024,
        hop_length: float = 0.05,
        window_size: float = 0.375,
        window_type: str = 'hann',
        device: str = 'cpu',
        **kwargs
    ):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        # Convert seconds to samples if needed, or store as is
        self.hop_length_s = hop_length
        self.window_size_s = window_size
        self.hop_length = int(hop_length * sample_rate)
        self.n_win = int(window_size * sample_rate)
        self.device = device

    def extract_features(self, signal):
        """
        Main feature extraction entry point for RadarApp.
        Handles Torch tensors and returns Torch tensors.
        """
        # Convert Torch to NumPy for librosa (which is NumPy based)
        if isinstance(signal, torch.Tensor):
            y = signal.cpu().numpy()
        else:
            y = signal
            
        mfcc = self.extract_mfcc(y)
        
        # Convert back to Torch
        return torch.from_numpy(mfcc).to(self.device)

    def extract_mfcc(self, signal: np.ndarray) -> np.ndarray:
        y = np.asarray(signal, dtype=np.float32)
        if y.ndim > 1:
            y = y.mean(axis=0)
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=int(self.sample_rate),
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        return mfcc

    def reconstruct_from_mfcc(self, mfcc: np.ndarray) -> np.ndarray:
        # Inverse MFCC to audio (rough approximation)
        y = librosa.feature.inverse.mfcc_to_audio(mfcc, sr=int(self.sample_rate))
        return y

    def extract_stft(self, signal: np.ndarray) -> np.ndarray:
        y = np.asarray(signal, dtype=np.float32)
        S = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)
        return np.abs(S)
