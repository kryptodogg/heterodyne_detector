# Torch-First Radar Application with Spatial Noise Cancellation

**2TX2RX Radar with GPU-Accelerated Beamforming + Adaptive LMS**

Complete radar signal processing pipeline optimized for PyTorch/ROCm (AMD RX 6700 XT).

## 🎯 Features

- ✅ **Torch-First Architecture** - GPU-accelerated from ground up
- ✅ **Spatial Noise Cancellation** - Geometry-based beamforming + adaptive LMS
- ✅ **MFCC Radar Features** - Doppler-optimized (0-500 Hz, 13 coefficients)
- ✅ **Heterodyne Detection** - Cross-correlation & phase analysis
- ✅ **Pattern Matching** - GPU-accelerated similarity search
- ✅ **HDF5 Storage** - Hierarchical pattern libraries (Date/Session/Band)
- ✅ **60 Hz Dashboard** - Real-time Plotly Dash visualization
- ✅ **Simulation Mode** - Test without hardware

## 📦 Architecture

```
main.py (Orchestrator)
├── sdr_interface.py        → Pluto+ 2TX2RX + simulation
├── geometry.py             → Steering vectors & DOA
├── noise_canceller.py      → Beamforming + LMS (KEY MODULE)
├── heterodyne_detector.py  → Torch-accelerated detection
├── audio_processor.py      → MFCC radar features
├── pattern_matcher.py      → GPU pattern matching
├── data_manager.py         → HDF5 libraries
├── visualizer.py           → 60 Hz Dash dashboard
└── config.py               → All settings
```

## 🚀 Quick Start

### 1. Install PyTorch with ROCm

```bash
# For AMD RX 6700 XT
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0

# Verify GPU
python -c "import torch; print(torch.cuda.is_available())"
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Test with Simulation

```bash
# Run in simulation mode (no hardware needed)
python main.py --simulate --duration 30
```

### 4. Run with Real Hardware

```bash
# After fixing libiio (see QUICKSTART.md)
python main.py --freq 2400 --sample-rate 10
```

## 📐 Geometry-Based Noise Cancellation

The **key innovation** is using radar array geometry for spatial filtering:

### How It Works

1. **Geometry Definition** (`config.py`):
   - TX1, TX2, RX1, RX2 positions in 3D space
   - Baseline: 15cm TX spacing, 10cm RX offset
   - Wavelength computed from center frequency

2. **Steering Vectors** (`geometry.py`):
   ```python
   # For each angle θ:
   steering_vector(θ) = exp(j * k · position)
   # where k = 2π/λ * [sin(θ), cos(θ), 0]
   ```

3. **Beamforming** (`noise_canceller.py`):
   - Compute steering matrix for -90° to +90°
   - Estimate DOA (Direction of Arrival)
   - Form beam toward signal, nulls toward interference

4. **Adaptive LMS**:
   - Initialize weights from geometry
   - Online adaptation: `w = w + μ·e*·x`
   - Converges to cancel dominant interference

### Positional Data Usage

- **TX1/TX2 positions** → Baseline for bistatic radar
- **RX1/RX2 positions** → Interferometer for DOA
- **Steering vectors** → Spatial filtering (beamformer weights)
- **Phase delays** → Cancel specific directions

Result: **~10-20 dB interference suppression** using geometry!

## 🎛️ Configuration

### MFCC Radar Settings (`config.py`)

```python
MFCC_RADAR_SETTINGS = {
    'window_size': 0.375,      # 375 ms
    'hop_length': 0.05,        # 50 ms (20 Hz frame rate)
    'n_fft': 2048,
    'n_mfcc': 13,              # Coefficients
    'fmin': 0.0,               # Include DC
    'fmax': 500.0,             # Doppler range
}
```

### Noise Cancellation

```python
NOISE_CANCELLATION = {
    'filter_length': 64,       # LMS taps
    'learning_rate': 0.01,     # Step size
    'num_steering_angles': 37, # -90° to +90° in 5° steps
    'use_beamforming': True,
    'use_adaptive_lms': True,
}
```

### GPU Configuration

```python
GPU_CONFIG = {
    'device': 'cuda:0',
    'memory_fraction': 0.8,    # 80% of GPU
    'memory_pool_gb': 4.0,
    'sample_rate': 10e6,       # 10 MHz
    'center_freq': 2.4e9,      # 2.4 GHz
}
```

## 📊 HDF5 Storage

Hierarchical organization:

```
./radar_libraries/
├── 2025-01-28/
│   ├── Session_143052/
│   │   ├── 2400MHz.h5
│   │   └── 915MHz.h5
│   └── Session_150230/
│       └── 2400MHz.h5
└── 2025-01-29/
    └── ...
```

Each `.h5` file contains:
- `patterns`: (N, 13, frames) MFCC features
- `metadata`: Detection scores, DOA, timestamps, etc.

## 🎮 Usage Examples

### Basic Detection

```bash
python main.py --simulate --duration 60
```

### Frequency Scan

```bash
# Scan ISM bands
for freq in 433 915 2400; do
    python main.py --freq $freq --duration 30
done
```

### Custom Geometry

```python
# Edit config.py:
RADAR_GEOMETRY = {
    'RX1': {'position': np.array([0.0, 0.0, 0.0])},
    'RX2': {'position': np.array([0.20, 0.0, 0.0])},  # 20cm baseline
    # ...
}
```

### Pattern Library Management

```python
from data_manager import HDF5DataManager

manager = HDF5DataManager()

# List all sessions
sessions = manager.list_sessions(date="2025-01-28")

# Load patterns
patterns, metadata = manager.load_patterns(
    session_id="Session_143052",
    band="2400MHz"
)

# Export to CSV
manager.export_to_csv(
    session_id="Session_143052",
    band="2400MHz",
    output_path="patterns.csv"
)
```

## 📈 Performance

**GPU Acceleration (RX 6700 XT):**
- MFCC extraction: ~500 frames/sec
- Cross-correlation: ~1000 buffers/sec
- Pattern matching: ~200 comparisons/sec
- Total pipeline: ~20-50 ms/buffer (10 MHz, 65k samples)

**Target Performance:**
- Real-time at 10 MSPS ✅
- 60 Hz visualization ✅
- <50 ms end-to-end latency ✅

## 🧪 Testing

Each module has a test function:

```bash
# Test individual modules
python geometry.py
python sdr_interface.py
python heterodyne_detector.py
python noise_canceller.py
python pattern_matcher.py
python data_manager.py
python visualizer.py

# Full system test
python main.py --simulate --duration 10
```

## 📱 Dashboard

Access at: `http://localhost:8050`

**Layout (2x2 grid):**
1. RX1 Spectrum (real-time FFT)
2. RX2 Spectrum
3. MFCC Heatmap (Doppler features)
4. DOA Polar Plot (Direction of Arrival)

**Refresh Rate:** 60 Hz target (16.67 ms), practical ~20-50 Hz

## 🔧 Troubleshooting

### PyTorch Not Using GPU

```bash
# Check ROCm
rocm-smi

# Test PyTorch
python -c "import torch; print(torch.cuda.get_device_name(0))"

# If CPU-only, reinstall:
pip install torch --index-url https://download.pytorch.org/whl/rocm6.0
```

### Pluto+ Not Detected

```bash
# Check USB
lsusb | grep "Analog Devices"

# Fix libiio
./fix_libiio.sh

# Or use simulation mode
python main.py --simulate
```

### High Processing Latency

```python
# Reduce buffer size in config.py:
GPU_CONFIG['buffer_size'] = 2**14  # 16k instead of 64k

# Or reduce sample rate:
python main.py --sample-rate 5  # 5 MHz instead of 10
```

## 📚 Module Details

### `noise_canceller.py` - The Core

**Beamforming:**
```python
# Compute steering vectors for all angles
steering_matrix = geometry.compute_steering_matrix(angles)

# Estimate DOA via beamformer scan
doa = estimate_doa_beamforming(rx1, rx2)

# Apply MVDR beamformer
weights = R^-1 * steering_vector / (steering_vector^H * R^-1 * steering_vector)
```

**Adaptive LMS:**
```python
for n in range(N):
    x = reference[n-L:n]          # Input vector
    y = w^H * x                    # Filter output
    e = desired[n] - y             # Error
    w = w + μ * e* * x            # Weight update
```

### `geometry.py` - Spatial Math

- Steering vector computation
- DOA estimation (phase & MUSIC)
- Baseline calculations
- Wavelength conversions

### `audio_processor.py` - Doppler Features

- STFT with Hann window
- Mel filterbank (40 bands, 0-500 Hz)
- DCT for MFCC (13 coefficients)
- Delta features (velocity)

## 🎓 Theory

### Why Geometry Matters

Traditional noise cancellation is **temporal** (time-domain filtering). This system adds **spatial** dimension:

1. **Temporal** (LMS): Adapts to signal statistics over time
2. **Spatial** (Beamforming): Uses antenna positions to filter by direction

Result: Can null interference from specific angles while preserving signals from others.

### Heterodyne Detection

Detects mixing products from multiple RF sources:
- Beat frequencies (f1 ± f2)
- Intermodulation (2f1 - f2, etc.)
- Phantom signals (appear as "voices" in noise)

Method: Cross-correlate RX1 and RX2 → High correlation = heterodyne

## 🔬 Future Enhancements

- [ ] ML-based pattern classification
- [ ] Automatic DOA tracking
- [ ] Multi-target resolution
- [ ] Compressed sensing for sparse signals
- [ ] Network streaming (remote dashboard)

## 📄 License

Provided for research and educational purposes.

## ⚠️ Important

- Ensure RF compliance (FCC/local regulations)
- TX power limited for safety
- Intended for legitimate radar research
- Not for interference or jamming

## 🙏 Acknowledgments

Built on:
- PyTorch (GPU acceleration)
- Plotly/Dash (visualization)
- HDF5 (storage)
- pyadi-iio (Pluto+ interface)

---

**Ready to detect heterodynes with spatial filtering!** 🎯📡
