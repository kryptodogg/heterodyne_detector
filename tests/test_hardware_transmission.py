#!/usr/bin/env python3
"""
Hardware Verification: 2TX2RX Transmission Test
Transmits unique pilot tones on TX1/TX2 and verifies simultaneous reception on RX1/RX2.
"""

import torch
import numpy as np
import time
import sys
import os

# Add parent dir to path for imports
sys.path.append(os.getcwd())

from sdr_interface import PlutoRadarInterface
from config import GPU_CONFIG

def run_transmission_test():
    print("="*60)
    print("📡 LIVE 2TX2RX HARDWARE TRANSMISSION TEST")
    print("="*60)

    # 1. Initialize Interface (Force Hardware)
    sdr = PlutoRadarInterface(
        sample_rate=GPU_CONFIG['sample_rate'],
        center_freq=GPU_CONFIG['center_freq'],
        simulate=False
    )

    if not sdr.connect():
        print("❌ FAILED: Could not connect to PlutoSDR hardware.")
        return

    try:
        # 2. Prepare Pilot Signals (Unique frequencies for identification)
        fs = sdr.sample_rate
        N = 2**16
        t = np.arange(N) / fs
        
        f_tx1 = 100e3  # 100 kHz offset
        f_tx2 = -200e3 # -200 kHz offset
        
        # Complex sine waves
        tx1_data = np.exp(1j * 2 * np.pi * f_tx1 * t).astype(np.complex64)
        tx2_data = np.exp(1j * 2 * np.pi * f_tx2 * t).astype(np.complex64)
        
        print(f"   Transmitting: TX1={f_tx1/1e3} kHz, TX2={f_tx2/1e3} kHz")
        
        # 3. Start Transmission (Cyclic buffer)
        # We need access to the raw sdr object for cyclic tx
        if hasattr(sdr.sdr, 'tx'):
            # Interleave if required by API, but standard pyadi-iio tx() 
            # handles list of channels
            sdr.sdr.tx([tx1_data * 10000, tx2_data * 10000]) 
            print("   TX Cyclic Buffer: Active")
        else:
            print("⚠️  Warning: SDR object does not support .tx() as expected")

        # 4. Capture and Analyze
        print("   Capturing signals...")
        time.sleep(0.5) # Let AGC/Synthesizers settle
        
        rx1, rx2 = sdr.rx()
        
        # FFT Analysis (CPU for verification stability)
        def get_peak_freq(signal, fs):
            fft = np.abs(np.fft.fft(signal))
            freqs = np.fft.fftfreq(len(signal), 1/fs)
            peak_idx = np.argmax(fft[1:]) + 1 # Skip DC
            return freqs[peak_idx], 20 * np.log10(fft[peak_idx] / len(signal))

        freq1, pwr1 = get_peak_freq(rx1, fs)
        freq2, pwr2 = get_peak_freq(rx2, fs)

        print("\n📊 Results:")
        print(f"   RX1 Peak: {freq1/1e3:+.2f} kHz | Power: {pwr1:.1f} dB")
        print(f"   RX2 Peak: {freq2/1e3:+.2f} kHz | Power: {pwr2:.1f} dB")

        # 5. Synchronization Check
        # Cross-correlation peak phase
        corr = np.correlate(rx1, rx2, mode='same')
        phase_diff = np.angle(corr[len(corr)//2]) * 180 / np.pi
        print(f"   Phase Offset (RX1-RX2): {phase_diff:.2f}°")

        # Validation
        success = True
        if abs(freq1 - f_tx1) > 5e3: 
            print("❌ FAIL: RX1 did not see TX1 pilot tone")
            success = False
        if abs(freq2 - f_tx1) > 5e3 and abs(freq2 - f_tx2) > 5e3:
            print("❌ FAIL: RX2 signal is missing expected pilots")
            success = False
            
        if success:
            print("\n✅ SUCCESS: 2TX2RX Hardware Link Verified")
        else:
            print("\n❌ FAILED: Hardware link issues detected")

    except Exception as e:
        print(f"❌ ERROR during test: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\n🛑 Closing SDR...")
        sdr.close()

if __name__ == "__main__":
    run_transmission_test()
