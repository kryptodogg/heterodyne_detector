#!/usr/bin/env python3
"""
Pluto+ Radar/Sonar Interface (2TX2RX)
Async-first abstraction for GPU-accelerated beacon probing.

Supports: real Pluto USB via Analog Devices libiio, or simulated mode.
Data path: rx() async generator yields (rx1, rx2) as numpy arrays (complex64).
"""

import asyncio
import numpy as np
import time
from typing import Tuple, Optional, AsyncIterator

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

    async def connect(self) -> bool:
        """Async SDR connection - non-blocking initialization"""
        if self.simulate:
            print("== SIMULATION MODE: PlutoRadarInterface ready ==")
            await asyncio.sleep(0)  # Yield control
            return True

        # Real hardware path (run in executor to avoid blocking)
        if adi is None:
            print("❌ PyADI not installed; cannot connect to Pluto+")
            return False

        # Hardware initialization can block, so run in thread pool
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, self._connect_hardware)
            # Set Gains
            self.sdr.gain_control_mode_chan0 = 'manual'
            self.sdr.gain_control_mode_chan1 = 'manual'
            self.sdr.rx_hardwaregain_chan0 = 50
            self.sdr.rx_hardwaregain_chan1 = 50
            
            # --- Pilot Tone Transmission (2TX) ---
            # Generate unique tones for TX1 and TX2
            N_tx = 2**14
            t_tx = np.arange(N_tx) / self.sample_rate
            tx1_pilot = np.exp(1j * 2 * np.pi * 100e3 * t_tx).astype(np.complex64)
            tx2_pilot = np.exp(1j * 2 * np.pi * -200e3 * t_tx).astype(np.complex64)
            
            # Send to cyclic buffer
            self.sdr.tx([tx1_pilot * 10000, tx2_pilot * 10000])
            
            self.channels = 2
            print("✅ Pluto+ 2TX2RX Hardware Interface Connected (Pilots Active)")
            return True
        except Exception as e:
            print(f"❌ Pluto connection failed: {e}")
            return False

    def _connect_hardware(self):
        """Synchronous hardware connection (called in executor)"""
        # Use network URI for better reliability
        uri = "ip:192.168.2.1"
        print(f"Connecting to PlutoSDR at {uri}...")

        # Use generic AD9361 class for full 2TX2RX access
        self.sdr = adi.ad9361(uri=uri)

        # Configure global parameters
        self.sdr.sample_rate = int(self.sample_rate)

        # RX Configuration
        self.sdr.rx_lo = int(self.center_freq)
        self.sdr.rx_enabled_channels = [0, 1]  # RX1 and RX2
        self.sdr.rx_buffer_size = 2**16

        # TX Configuration (Standard 2TX setup)
        self.sdr.tx_lo = int(self.center_freq)
        self.sdr.tx_enabled_channels = [0, 1]  # TX1 and TX2
        self.sdr.tx_cyclic_buffer = True  # Continuous beacon probing

        # Set Gains
        self.sdr.gain_control_mode_chan0 = 'manual'
        self.sdr.gain_control_mode_chan1 = 'manual'
        self.sdr.rx_hardwaregain_chan0 = 50
        self.sdr.rx_hardwaregain_chan1 = 50

        self.channels = 2

    async def rx(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Async receive - yields control while waiting for samples.
        Returns (rx1, rx2) tuple without blocking event loop.
        """
        if self.simulate:
            # Simulate slight delay for realistic async behavior
            await asyncio.sleep(0.001)  # 1ms simulated acquisition
            return self._generate_synthetic_signals()
        else:
            # Real hardware read (run in executor to avoid blocking)
            loop = asyncio.get_event_loop()
            samples = await loop.run_in_executor(None, self._rx_hardware)

            if isinstance(samples, list) and len(samples) >= 2:
                # Proper 2RX return
                rx1 = np.asarray(samples[0], dtype=np.complex64)
                rx2 = np.asarray(samples[1], dtype=np.complex64)
                return rx1, rx2
            elif isinstance(samples, np.ndarray) and samples.ndim > 1:
                # 2D array return
                return samples[0], samples[1]
            else:
                # Fallback: return as single channel data duplicated
                s = np.asarray(samples, dtype=np.complex64)
                return s, s

    def _rx_hardware(self):
        """Synchronous hardware receive (called in executor)"""
        return self.sdr.rx()

    def _generate_synthetic_signals(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic signals with proper buffer size"""
        # Use standard buffer size (2^16 = 65536 samples)
        buffer_size = 2**16

        t = np.linspace(
            self.time,
            self.time + buffer_size / self.sample_rate,
            buffer_size,
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

    async def close(self):
        """Async cleanup - non-blocking shutdown"""
        if self.sdr is not None:
            loop = asyncio.get_event_loop()
            try:
                await loop.run_in_executor(None, self.sdr.close)
            except Exception:
                pass

    def close_sync(self):
        """Synchronous fallback for compatibility"""
        if self.sdr is not None:
            try:
                self.sdr.close()
            except Exception:
                pass
