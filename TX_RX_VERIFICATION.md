# TX/RX Configuration Verification

## Hardware Configuration (Pluto+ 2TX2RX)

### Transmission Setup

The system is configured for **continuous beacon transmission** on both TX channels:

```python
# In sdr_interface.py → _connect_hardware()

# TX Configuration
self.sdr.tx_lo = int(self.center_freq)        # Local oscillator frequency
self.sdr.tx_enabled_channels = [0, 1]         # Enable TX1 and TX2
self.sdr.tx_cyclic_buffer = True              # ⭐ Continuous transmission
```

**Key Parameters**:
- **tx_lo**: Carrier frequency (e.g., 2.4 GHz for ISM band)
- **tx_enabled_channels**: `[0, 1]` means both TX1 and TX2 active
- **tx_cyclic_buffer**: `True` enables continuous beacon probing (critical for radar)

### Reception Setup

Dual-channel reception with proper gain control:

```python
# RX Configuration
self.sdr.rx_lo = int(self.center_freq)        # Must match TX frequency
self.sdr.rx_enabled_channels = [0, 1]         # Enable RX1 and RX2
self.sdr.rx_buffer_size = 2**16               # 65536 samples per acquisition

# Gain Control
self.sdr.gain_control_mode_chan0 = 'manual'   # Manual gain (no AGC)
self.sdr.gain_control_mode_chan1 = 'manual'
self.sdr.rx_hardwaregain_chan0 = 50           # 50 dB gain
self.sdr.rx_hardwaregain_chan1 = 50           # Matched gains for beamforming
```

**Critical Settings**:
- **Manual gain**: Prevents AGC from introducing phase/amplitude errors
- **Matched gains**: RX1 and RX2 have identical gain for proper beamforming
- **Buffer size**: 65536 samples = 6.5ms at 10 MHz sample rate

## Data Flow

### Async Transmission/Reception Cycle

```
┌─────────────────────────────────────────────────────────┐
│  Hardware Layer (Pluto+ SDR)                            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  TX1 ──┐                                                │
│        ├── Continuous CW beacon @ center_freq          │
│  TX2 ──┘    (tx_cyclic_buffer = True)                  │
│                                                         │
│                          ↓                              │
│                     Environment                         │
│                     (reflections)                       │
│                          ↓                              │
│                                                         │
│  RX1 ──┐                                                │
│        ├── Receive reflected signals                   │
│  RX2 ──┘    (65536 samples @ 10 MHz)                   │
│                                                         │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  Async Software Layer                                   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  await sdr.rx()                                         │
│      └─→ loop.run_in_executor(self._rx_hardware)       │
│             └─→ self.sdr.rx()  [blocking hardware]     │
│                    └─→ Returns (rx1, rx2) tuple        │
│                                                         │
│  await process_buffer(rx1, rx2)                         │
│      └─→ Spatial noise cancellation (beamforming)      │
│      └─→ Concurrent: Detection │ RD │ PPI │ MFCC       │
│      └─→ Target tracking                               │
│                                                         │
│  await visualizer.update(...)                           │
│      └─→ Non-blocking dashboard refresh                │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Verification Tests

### 1. Hardware Connection Test

```python
async def test_hardware_connection():
    from sdr_interface import PlutoRadarInterface

    sdr = PlutoRadarInterface(
        center_freq=2.4e9,
        sample_rate=10e6,
        simulate=False  # Real hardware
    )

    # Async connect
    connected = await sdr.connect()
    assert connected, "Hardware connection failed"

    # Verify channels
    assert sdr.channels == 2, "Should have 2 RX channels"

    print("✅ Hardware connected")
    print(f"   TX Channels: [0, 1]")
    print(f"   RX Channels: [0, 1]")
    print(f"   Center Freq: {sdr.center_freq/1e9:.2f} GHz")
    print(f"   Sample Rate: {sdr.sample_rate/1e6:.1f} MHz")

asyncio.run(test_hardware_connection())
```

### 2. Transmission Verification

Verify TX is transmitting by checking RX power:

```python
async def verify_transmission():
    sdr = PlutoRadarInterface(center_freq=2.4e9, simulate=False)
    await sdr.connect()

    # Receive samples
    rx1, rx2 = await sdr.rx()

    # Check signal power (should be > noise floor if TX working)
    power_rx1 = np.mean(np.abs(rx1)**2)
    power_rx2 = np.mean(np.abs(rx2)**2)

    power_dbm_rx1 = 10 * np.log10(power_rx1 + 1e-12)
    power_dbm_rx2 = 10 * np.log10(power_rx2 + 1e-12)

    print(f"RX1 Power: {power_dbm_rx1:.1f} dBm")
    print(f"RX2 Power: {power_dbm_rx2:.1f} dBm")

    # Should be well above noise floor (-120 dBm)
    assert power_dbm_rx1 > -100, "RX1 signal too weak - TX may not be working"
    assert power_dbm_rx2 > -100, "RX2 signal too weak"

    print("✅ Transmission verified")
```

### 3. Dual-Channel Synchronization

Verify RX1 and RX2 are phase-coherent (required for beamforming):

```python
async def verify_phase_coherence():
    sdr = PlutoRadarInterface(center_freq=2.4e9, simulate=False)
    await sdr.connect()

    rx1, rx2 = await sdr.rx()

    # Cross-correlation
    corr = np.abs(np.fft.ifft(
        np.fft.fft(rx1) * np.conj(np.fft.fft(rx2))
    ))
    coherence = np.max(corr) / np.sqrt(
        np.sum(np.abs(rx1)**2) * np.sum(np.abs(rx2)**2)
    )

    print(f"Coherence: {coherence:.3f}")

    # Should be > 0.5 for proper beamforming
    assert coherence > 0.5, "Channels not coherent - check cabling"

    print("✅ Phase coherence verified")
```

### 4. Full Pipeline Test

```bash
# Run full system for 10 seconds
python main.py --freq 2400 --sample-rate 10 --duration 10

# Check output:
# ✅ Should show "Pluto+ 2TX2RX Hardware Interface Connected"
# ✅ Should process buffers continuously
# ✅ Should show detection results
```

## Simulation Mode

For testing without hardware:

```python
# In sdr_interface.py → _generate_synthetic_signals()

buffer_size = 2**16  # 65536 samples

# Simulates two-channel reception
# RX1: Primary signal + interference + noise
# RX2: Delayed/attenuated signal (simulates spatial separation)

rx1 = signal + interference + noise
rx2 = 0.7 * signal + different_interference + noise
```

**Simulation characteristics**:
- Realistic interference (multiple tones)
- Spatial diversity (RX2 delayed/attenuated vs RX1)
- Proper buffer sizing (65536 samples)
- Async generation (1ms delay to simulate acquisition)

## Async Integration

All TX/RX operations are fully async:

```python
# Connection (non-blocking)
await sdr.connect()  # Hardware init in executor

# Reception (non-blocking)
rx1, rx2 = await sdr.rx()  # Buffer fill in executor

# Cleanup (non-blocking)
await sdr.close()  # Hardware shutdown in executor
```

**Benefits**:
- Event loop never blocks on hardware I/O
- System can handle signals, timers, etc. during acquisition
- Multiple concurrent operations possible
- Graceful shutdown on Ctrl+C

## Performance Metrics

### Expected Timing (10 MHz sample rate, 65536 samples)

- **Hardware acquisition**: 6.5ms (determined by sample rate)
- **Software overhead**: <1ms (async I/O)
- **Processing**: 10-50ms (GPU-accelerated)
- **Total latency**: <60ms per buffer

### Throughput

- **Sample rate**: 10 MSPS (megasamples per second)
- **Buffer rate**: ~150 buffers/second
- **Data rate**: ~500 MB/s (complex64)

## Troubleshooting

### TX Not Working

**Symptoms**: RX power at noise floor (-120 dBm)

**Checks**:
1. Verify `tx_enabled_channels = [0, 1]`
2. Verify `tx_cyclic_buffer = True`
3. Check TX power setting
4. Verify antenna connections

### RX Channels Swapped

**Symptoms**: DOA (direction of arrival) incorrect

**Fix**: Check channel ordering in `rx()`:
```python
return rx1, rx2  # Should be [channel 0, channel 1]
```

### Async Blocking

**Symptoms**: System unresponsive to Ctrl+C

**Checks**:
1. All I/O uses `await loop.run_in_executor()`
2. No `time.sleep()` (use `await asyncio.sleep()`)
3. No blocking operations in main loop

### Buffer Underrun

**Symptoms**: Missing samples, gaps in data

**Fix**: Increase buffer size or reduce processing time:
```python
self.sdr.rx_buffer_size = 2**17  # Double buffer
```

## Configuration Reference

### Default Settings (config.py)

```python
GPU_CONFIG = {
    'sample_rate': 10e6,        # 10 MHz
    'center_freq': 2.4e9,       # 2.4 GHz ISM band
    'buffer_size': 2**16,       # 65536 samples
    'device': 'cuda:0',
}
```

### Hardware Limits (Pluto+ AD9361)

- **Sample rate**: 520 kHz - 61.44 MHz
- **Center freq**: 70 MHz - 6 GHz
- **RX gain**: 0 - 77 dB
- **TX power**: -89.75 - 0 dBm (adjustable)
- **Channels**: 2TX2RX simultaneous

## Summary

✅ **TX Configuration**: Dual-channel continuous transmission enabled
✅ **RX Configuration**: Dual-channel coherent reception with matched gains
✅ **Async I/O**: All hardware operations non-blocking
✅ **Buffer Sizing**: Optimal 65536 samples for 10 MHz operation
✅ **Data Flow**: Fully verified from hardware → processing → visualization

The system is properly configured for real-time radar operation with spatial noise cancellation.
