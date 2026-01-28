#!/usr/bin/env python3
"""
Pluto+ Radar/Sonar Interface (2TX2RX)
Minimal abstraction for GPU-accelerated beacon probing.

Supports: real Pluto USB via Analog Devices libiio, or simulated mode.
Data path: rx() returns (rx1, rx2) as numpy arrays (complex64).
"""

import numpy as np
import time
from typing import Tuple, Optional

try:
    import adi  # PyADI-IO for Pluto+ (IIO API)
except Exception:
    adi = None  # allow simulate path


class PlutoRadarInterface:
    def __init__(
        self,
        center_freq: float = 2.45e9,
        bandwidth: float = 20e6,
        sample_rate: float = 2.4e6,
        simulate: bool = True,
        device: str = 'cpu'
    ):
        self.center_freq = center_freq
        self.bandwidth = bandwidth
        self.sample_rate = sample_rate
        self.simulate = simulate
        self.device = device
        self.sdr = None
        self.time = 0.0
        self.channels = 2  # 2 RX channels (2TX2RX capable)

    def connect(self) -> bool:
        if self.simulate:
            print("== SIMULATION MODE: PlutoRadarInterface ready ==")
            return True
        # Real hardware path
        if adi is None:
            print("❌ PyADI not installed; cannot connect to Pluto+")
            return False
        try:
            self.sdr = adi.Pluto("usb:1.100.5")
            self.sdr.sample_rate = int(self.sample_rate)
            self.sdr.rx_lo = int(self.center_freq)
            self.sdr.rx_buffer_size = 2**16
            self.channels = 2
            return True
        except Exception as e:
            print(f"Pluto connection failed: {e}")
            return False

    def rx(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.simulate:
            return self._generate_synthetic_signals()
        else:
            # Real hardware read
            samples = self.sdr.rx()  # returns numpy array with two channels?
            if isinstance(samples, (list, tuple)) and len(samples) >= 2:
                return samples[0], samples[1]
            else:
                # Fallback: return as single channel data duplicated
                return samples, samples

    def _generate_synthetic_signals(self) -> Tuple[np.ndarray, np.ndarray]:
        t = np.linspace(
            self.time,
            self.time + self.bandwidth / self.sample_rate,
            int(self.sample_rate * (self.bandwidth / self.sample_rate)),
            endpoint=False,
        )
        self.time = t[-1] if len(t) > 0 else self.time

        # Simple simulated complex sine signals with interference
        f1 = 10e3  # 10 kHz carrier offset
        f2 = 15e3  # 15 kHz interference
        s1 = np.exp(1j * 2 * np.pi * f1 * t)
        s2 = 0.5 * np.exp(1j * 2 * np.pi * f2 * t)
        rx1 = (
            s1 + s2 + 0.1 * (np.random.randn(len(t)) + 1j * np.random.randn(len(t)))
        ).astype(np.complex64)
        rx2 = (
            0.7 * s1
            + 0.6 * np.exp(1j * 2 * np.pi * 5e3 * t)
            + 0.1 * (np.random.randn(len(t)) + 1j * np.random.randn(len(t)))
        ).astype(np.complex64)
        return rx1, rx2

    def close(self):
        if self.sdr is not None:
            try:
                self.sdr.close()
            except Exception:
                pass
