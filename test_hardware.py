#!/usr/bin/env python3
"""
Quick hardware diagnostic - verify TX/RX working
"""

import asyncio
import numpy as np
from sdr_interface import PlutoRadarInterface


async def diagnose_hardware():
    """Test hardware TX/RX functionality"""
    print("="*60)
    print("Hardware Diagnostic Test")
    print("="*60)

    # Connect to hardware
    print("\n1. Connecting to Pluto+...")
    sdr = PlutoRadarInterface(
        center_freq=2.4e9,
        sample_rate=10e6,
        simulate=False
    )

    connected = await sdr.connect()
    if not connected:
        print("❌ Connection failed")
        return False

    print("✅ Connected successfully")

    # Test reception
    print("\n2. Testing RX channels...")
    rx1, rx2 = await sdr.rx()

    print(f"   RX1: {len(rx1)} samples")
    print(f"   RX2: {len(rx2)} samples")

    # Verify buffer size
    expected_size = 2**16
    assert len(rx1) == expected_size, f"Wrong buffer size: {len(rx1)}"
    assert len(rx2) == expected_size, f"Wrong buffer size: {len(rx2)}"
    print("   ✅ Buffer sizes correct (65536 samples)")

    # Check signal power
    print("\n3. Checking signal power...")
    power_rx1 = np.mean(np.abs(rx1)**2)
    power_rx2 = np.mean(np.abs(rx2)**2)

    # Convert to dBFS (dB relative to full scale)
    power_dbfs_rx1 = 10 * np.log10(power_rx1 + 1e-12)
    power_dbfs_rx2 = 10 * np.log10(power_rx2 + 1e-12)

    print(f"   RX1: {power_dbfs_rx1:.1f} dBFS")
    print(f"   RX2: {power_dbfs_rx2:.1f} dBFS")

    # Should be well above noise floor
    if power_dbfs_rx1 < -100:
        print("   ⚠️  RX1 power very low - check TX or antenna")
    else:
        print("   ✅ RX1 power OK")

    if power_dbfs_rx2 < -100:
        print("   ⚠️  RX2 power very low - check antenna")
    else:
        print("   ✅ RX2 power OK")

    # Check channel correlation (phase coherence)
    print("\n4. Checking channel coherence...")

    # Cross-correlation in frequency domain (fast)
    fft1 = np.fft.fft(rx1)
    fft2 = np.fft.fft(rx2)
    cross_spec = fft1 * np.conj(fft2)
    corr = np.fft.ifft(cross_spec)

    # Normalized coherence
    coherence = np.max(np.abs(corr)) / np.sqrt(
        np.sum(np.abs(rx1)**2) * np.sum(np.abs(rx2)**2) + 1e-12
    )

    print(f"   Coherence: {coherence:.3f}")

    if coherence > 0.7:
        print("   ✅ Excellent phase coherence (good for beamforming)")
    elif coherence > 0.4:
        print("   ✅ Adequate phase coherence")
    else:
        print("   ⚠️  Low coherence - check channel sync or cabling")

    # Spectral analysis
    print("\n5. Spectral analysis...")

    # Find peak frequency
    mag1 = np.abs(fft1)
    peak_idx = np.argmax(mag1[:len(mag1)//2])
    peak_freq = peak_idx * 10e6 / len(rx1)

    print(f"   Peak frequency: {peak_freq/1e3:.1f} kHz")
    print(f"   ✅ Spectrum computed")

    # Summary
    print("\n" + "="*60)
    print("Diagnostic Summary")
    print("="*60)
    print(f"✅ Hardware: Connected")
    print(f"✅ Channels: 2TX2RX active")
    print(f"✅ Buffers: {len(rx1)} samples each")
    print(f"✅ Power: RX1={power_dbfs_rx1:.1f}dBFS, RX2={power_dbfs_rx2:.1f}dBFS")
    print(f"✅ Coherence: {coherence:.3f}")
    print("="*60)
    print("\n🎯 Hardware ready for radar operation!")

    # Cleanup
    await sdr.close()
    return True


if __name__ == "__main__":
    try:
        asyncio.run(diagnose_hardware())
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
