# Configuration Management for Radar Processing

## Bluetooth Configuration

### Ubuntu Bluetooth Stack Implementation

Instead of using PyBluez, the system now leverages Ubuntu's native Bluetooth stack for communication with HC-05 and ESP32 devices. This approach offers better system integration and reliability.

#### Key Configuration Parameters

1. **Device Address**: MAC address of the target Bluetooth device
2. **RFCOMM Channel**: Typically channel 1 for serial communication
3. **Connection Timeout**: Time to wait for connection establishment
4. **Pairing Mode**: Automatic or manual pairing

#### Configuration Example

```python
BLUETOOTH_CONFIG = {
    'device_address': 'AA:BB:CC:DD:EE:FF',  # Replace with actual device MAC
    'rfcomm_channel': 1,
    'connection_timeout': 10,  # seconds
    'auto_pair': True,
    'error_correction': True
}
```

#### System Dependencies

The following system packages are required:

```bash
sudo apt-get install bluetooth bluez-tools rfcomm
```

#### Permissions

The user running the radar processing system needs to be in the `dialout` group to access RFCOMM devices:

```bash
sudo usermod -a -G dialout $USER
```

## Radar Geometry Configuration

Defines physical antenna positions and derived metrics driving beamforming:

```python
RADAR_GEOMETRY = {
    'TX1': {
        'position': [0.0, 0.0, 0.0],      # (x, y, z) in meters
        'gain': 0.0,                       # dBi
        'phase_offset': 0.0                # radians
    },
    'TX2': {
        'position': [0.05, 0.0, 0.0],     # 5 cm horizontal separation
        'gain': 0.0,
        'phase_offset': 0.0
    },
    'RX1': {
        'position': [0.0, 0.0, 0.0],
        'gain': 0.0,
        'phase_offset': 0.0
    },
    'RX2': {
        'position': [0.05, 0.0, 0.0],     # Baseline = 5 cm
        'gain': 0.0,
        'phase_offset': 0.0
    },
    'wavelength': None,                   # Auto-computed from center_freq
    'center_freq': 2.4e9                  # Hz (2.4 GHz)
}
```

**Impact on Beamforming:**
- `RX1` and `RX2` positions determine steering vector phases: `phase = k * baseline * sin(angle)`
- **Smaller baseline** → finer angular resolution but lower DOA precision
- **Larger baseline** → coarser resolution but better localization
- For 2.4 GHz (λ ≈ 0.125 m): 5 cm baseline provides ~2.5° resolution

## GPU Configuration

Memory and compute resource allocation:

```python
GPU_CONFIG = {
    'sample_rate': 10e6,           # 10 MHz ADC sampling
    'center_freq': 2.4e9,          # 2.4 GHz carrier
    'buffer_size': 10000,          # Samples per processing frame
    'memory_fraction': 0.8,        # Max 80% of GPU memory
    'device': 'cuda:0',            # GPU device index
    'dtype': torch.complex64,      # Single precision (float32 complex)
    'pin_memory': True             # Enable DMA for serial ingress
}
```

**Memory Calculation:**
```
Per buffer:
  - RX1: 10k samples × 8 bytes (complex64) = 80 KB
  - RX2: 80 KB
  - Intermediate (FFT): 10k × 8 bytes = 80 KB
  - Weights (filter): 100 taps × 8 bytes × 2 ch = 1.6 KB
  - Total per frame: ~260 KB

Full pipeline (5-10 buffers queued):
  - Working memory: ~2-5 MB
  - GPU capacity: RX 6700 XT has 12 GB
  - Safe allocation: 2-4 GB for 1000+ frames/sec
```

**Tuning:**
- Increase `buffer_size` for better FFT efficiency (powers of 2)
- Decrease if experiencing VRAM exhaustion
- Monitor with `torch.cuda.memory_allocated()`

## Signal Processing Parameters

### Heterodyne Detection

```python
HETERODYNE_CONFIG = {
    'threshold': 0.7,              # Detection score [0-1]
    'frequency_offset_range': [-100e3, 100e3],  # ±100 kHz
    'window_function': 'hann',     # FFT windowing
    'fft_size': 8192,              # 2x zero-padding
    'voice_freq_band': [100, 4000] # Hz (speech band)
}
```

**Explanation:**
- `threshold`: Varies from ~0.5 (low specificity, many false alarms) to 0.95 (high specificity, misses weak targets)
- `frequency_offset_range`: Doppler shift depends on target velocity
  - ±100 kHz at 2.4 GHz → ±0.33 m/s velocity resolution
- `fft_size`: Larger FFT provides better frequency resolution but increases latency

### Noise Cancellation

```python
NOISE_CANCELLATION = {
    'filter_length': 128,          # LMS tap count
    'learning_rate': 0.01,         # μ parameter [0.001-0.1]
    'beamformer_type': 'mvdr',     # 'mvdr' or 'lcmv'
    'angle_resolution': 1.0,       # Degrees (sweep 0-180)
    'regularization': 1e-3,        # Diagonal loading for R matrix
    'num_angles': 180              # Beamformer scan points
}
```

**Learning Rate Tuning:**
```
μ too small (0.001):  Slow convergence, good stability
μ medium (0.01):      Balanced (recommended)
μ large (0.1):        Fast convergence, risk of divergence
μ too large (>0.2):   Unstable, oscillating weights
```

**Regularization (Diagonal Loading):**
```
R_regularized = R + λ * I

λ = 1e-3:  Conservative, stable beamformer
λ = 0:     Pure MVDR, risks singular matrix
λ = 0.1:   Aggressively stable, wider nulls
```

### Range-Doppler Processing

```python
RANGE_DOPPLER = {
    'fft_size': 256,               # Range dimension FFT
    'doppler_size': 128,           # Velocity dimension FFT
    'range_resolution': 0.3,       # Meters (c / 2 / BW)
    'velocity_resolution': 0.1,    # m/s (wavelength / 2 / T)
    'window': 'hamming'
}
```

**Derived Quantities:**
- **Range resolution**: Δr = c / (2 × BW)
  - For 10 MHz BW: Δr = 3×10⁸ / (2 × 10×10⁶) = **15 meters**
- **Velocity resolution**: Δv = λ / (2 × T)
  - For 100 ms integration: Δv = 0.125 / (2 × 0.1) = **0.625 m/s**

## Configuration Usage Pattern

```python
from config import (
    RADAR_GEOMETRY,
    NOISE_CANCELLATION,
    GPU_CONFIG,
    get_torch_geometry
)
import torch

# 1. Create geometry object (converts to GPU tensors)
geometry = get_torch_geometry(RADAR_GEOMETRY, device='cuda:0')

# 2. Initialize processors
detector = HeterodyneDetector(
    sample_rate=GPU_CONFIG['sample_rate'],
    device=torch.device('cuda:0')
)

canceller = SpatialNoiseCanceller(
    geometry=geometry,
    config=NOISE_CANCELLATION,
    device=torch.device('cuda:0')
)

# 3. Monitor and adjust
print(f"Wavelength: {geometry.wavelength:.4f} m")
print(f"Baseline: {geometry.baseline:.4f} m")
print(f"DOA resolution: {0.5/np.pi * geometry.wavelength/geometry.baseline:.2f}°")
```

## Performance Tuning Guidelines

### For Real-Time (Low Latency)

```python
GPU_CONFIG = {
    'buffer_size': 2048,           # Smaller buffers = lower latency
    'sample_rate': 5e6,            # Reduce sample rate if needed
}

RANGE_DOPPLER = {
    'fft_size': 128,               # Smaller FFT = faster
    'doppler_size': 64,
}

NOISE_CANCELLATION = {
    'filter_length': 64,           # Fewer taps = faster LMS
    'learning_rate': 0.05,         # Higher μ = faster convergence
}
```

**Expected latency**: 0.5-1 ms per buffer

### For High SNR (Maximum Noise Suppression)

```python
NOISE_CANCELLATION = {
    'filter_length': 256,          # More taps = better adaptation
    'learning_rate': 0.01,         # Slower, more stable
    'num_angles': 360,             # Finer angular scan
    'regularization': 1e-4,        # Less regularization
}

GPU_CONFIG = {
    'buffer_size': 20000,          # Larger buffers = more averaging
}
```

**Expected SNR improvement**: 8-12 dB (vs. 4-6 dB for low-latency tuning)

### For Power Constrained (Mobile/Embedded)

```python
GPU_CONFIG = {
    'memory_fraction': 0.3,        # Lower GPU clocks
    'dtype': torch.float16,        # Half precision (with care)
    'pin_memory': False,           # Skip DMA optimization
}

HETERODYNE_CONFIG = {
    'fft_size': 1024,              # Minimum practical size
}

NOISE_CANCELLATION = {
    'num_angles': 90,              # Coarse scan
}
```

**Tradeoff**: Reduced accuracy for ~30-40% power savings
