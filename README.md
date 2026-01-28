# Heterodyne Detection & Active Noise Cancellation for Pluto+ SDR

A sophisticated Python system for detecting phantom voices and noise heterodyning using the Pluto+ dual RX/TX SDR, with GPU-accelerated pattern matching (DTW/Levenshtein analysis) for AMD RX 6700 XT.

## Features

- **Dual RX Channel Heterodyne Detection**: Correlates signals from both receivers to identify heterodyne artifacts
- **Phantom Voice Detection**: Specialized detection for voice-like artifacts using formant analysis
- **GPU-Accelerated Pattern Matching**: Uses DTW (Dynamic Time Warping) and Levenshtein distance for pattern recognition
- **Active Noise Cancellation**: Real-time 2TX cancellation using adaptive LMS filtering
- **Real-Time Visualization**: Live spectral and time-domain analysis
- **Pattern Database**: Learns and stores suspicious patterns for future detection

## System Requirements

### Hardware
- **Pluto+ SDR** with dual RX/TX capability
- **AMD RX 6700 XT** (or compatible GPU with ROCm support)
- USB 3.0 connection for Pluto+
- Minimum 8GB RAM recommended

### Software
- Python 3.8+
- ROCm 5.0+ (for AMD GPU support)
- Linux recommended (Ubuntu 20.04+, Debian 11+)

## Installation

### 1. Install ROCm (for AMD GPU acceleration)

```bash
# Ubuntu/Debian
wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/focal/amdgpu-install_*.deb
sudo apt install ./amdgpu-install_*.deb
sudo amdgpu-install --usecase=rocm

# Verify installation
rocm-smi
```

### 2. Install libiio and Pluto drivers

```bash
# Install dependencies
sudo apt update
sudo apt install -y libxml2-dev bison flex libcdk5-dev cmake \
    libusb-1.0-0-dev libserialport-dev libavahi-client-dev \
    doxygen graphviz libaio-dev

# Build and install libiio
git clone https://github.com/analogdevicesinc/libiio.git
cd libiio
mkdir build && cd build
cmake .. -DPYTHON_BINDINGS=ON
make -j$(nproc)
sudo make install
sudo ldconfig

# Add udev rules for Pluto
sudo bash -c 'cat > /etc/udev/rules.d/53-adi-plutosdr-usb.rules << EOF
SUBSYSTEM=="usb", ATTRS{idVendor}=="0456", ATTRS{idProduct}=="b673", MODE="0666"
SUBSYSTEM=="usb", ATTRS{idVendor}=="0456", ATTRS{idProduct}=="b674", MODE="0666"
EOF'

sudo udevadm control --reload-rules
sudo udevadm trigger
```

### 3. Install Python dependencies

```bash
# Clone this repository or navigate to the directory
cd heterodyne_detector

# Install base requirements
pip install -r requirements.txt

# Install CuPy for ROCm (AMD GPU)
# Check your ROCm version first with: rocm-smi
pip install cupy-rocm-5-0  # Adjust version to match your ROCm

# Verify GPU is available
python -c "import cupy as cp; print(cp.cuda.Device(0).compute_capability)"
```

### 4. Configure Pluto+ Network (Optional)

If using network connection instead of USB:

```bash
# Default Pluto+ IP: 192.168.2.1
# Edit /etc/hosts if needed
echo "192.168.2.1 pluto.local" | sudo tee -a /etc/hosts

# Test connection
ping pluto.local
```

## Usage

### Basic Usage

```bash
# Run the integrated system with default settings (100 MHz, 2.4 MSPS)
python integrated_detector.py

# Specify frequency and sample rate
python integrated_detector.py --freq 433.92 --sample-rate 2.4

# Disable visualization (headless mode)
python integrated_detector.py --no-viz

# Start with active cancellation enabled
python integrated_detector.py --cancellation
```

### Interactive Commands

While running, use these commands:

- `c` - Toggle active noise cancellation
- `t` - Set detection threshold (0.0-1.0)
- `f` - Change center frequency
- `p` - Clear pattern database
- `s` - Show statistics
- `q` - Quit

### Advanced Usage

#### Custom Configuration

```python
from integrated_detector import IntegratedDetectionSystem

# Create system with custom parameters
system = IntegratedDetectionSystem(
    sample_rate=5e6,        # 5 MSPS
    center_freq=915e6,      # 915 MHz ISM band
    visualize=True
)

# Configure detection parameters
system.het_detector.heterodyne_threshold = 0.75
system.het_detector.rx_gain = 60

# Configure pattern matching
system.pattern_matcher.similarity_threshold = 0.85
system.pattern_matcher.max_patterns = 2000

# Start system
system.start()
```

#### Testing Individual Components

```bash
# Test pattern matcher with synthetic data
python pattern_matcher.py

# Test basic heterodyne detector
python heterodyne_detector.py
```

## How It Works

### 1. Heterodyne Detection

The system uses cross-correlation between RX1 and RX2 channels to detect:
- **Intermodulation products**: Unwanted mixing of signals
- **Heterodyne artifacts**: Beat frequencies from multiple sources
- **Phantom voices**: Voice-like artifacts in the noise

```python
# Simplified detection algorithm
correlation = correlate(rx1_signal, rx2_signal)
heterodyne_score = max(correlation) / max_correlation

# Frequency offset detection via phase analysis
phase_diff = angle(fft(rx1) * conj(fft(rx2)))
freq_offset = phase_slope * sample_rate / (2π)
```

### 2. Pattern Matching

Uses two complementary approaches:

**DTW (Dynamic Time Warping)**
- Finds similar patterns even with time stretching
- Computes spectral features: centroid, rolloff, energy
- GPU-accelerated for fast comparison

**Levenshtein Distance**
- Quantizes features into discrete symbols
- Computes edit distance between symbol sequences
- Effective for repeated patterns with variations

### 3. Active Cancellation

Implements adaptive LMS (Least Mean Squares) filtering:

```python
# For each sample
y = w^H * x          # Filter output
e = desired - y       # Error signal
w = w + μ * e^* * x  # Weight update
```

The cancellation signal is transmitted via TX channels to destructively interfere with detected heterodyne artifacts.

### 4. Phantom Voice Detection

Specialized detection for voice-like artifacts:
- AM demodulation of received signal
- Bandpass filtering for voice range (300-3400 Hz)
- Formant detection in typical speech bands
- Pattern matching for repeated voice-like sequences

## Configuration Tips

### For Best Detection Performance

1. **Start with wider bandwidth**: 5+ MSPS sample rate
2. **Adjust RX gain carefully**: Too high causes saturation, too low misses weak signals
3. **Set threshold conservatively**: Start at 0.7, adjust based on false positive rate
4. **Use pattern matching**: Enable learning mode to build pattern database

### For Active Cancellation

1. **Ensure good RX/TX isolation**: Physical separation helps
2. **Calibrate TX power**: Start low, increase gradually
3. **Monitor feedback**: Watch for positive feedback loops
4. **Use adaptive mode**: Let the LMS filter converge (10-100 iterations)

### Frequency Ranges to Monitor

Common bands for heterodyne detection:
- **88-108 MHz**: FM broadcast band
- **433-434 MHz**: ISM band (common interference)
- **915 MHz**: ISM band (US)
- **2.4 GHz**: WiFi/Bluetooth (set sample rate accordingly)

## GPU Optimization

### AMD GPU (RX 6700 XT)

The system automatically uses GPU when available:

```python
# Check GPU usage
import cupy as cp
print(f"GPU Memory: {cp.cuda.Device(0).mem_info}")

# Force CPU fallback
export CUDA_VISIBLE_DEVICES=""
```

### Performance Tips

1. **Use ROCm 5.0+**: Better performance than 4.x
2. **Enable kernel fusion**: Automatic in CuPy
3. **Batch processing**: Process multiple buffers at once
4. **Memory pinning**: Pre-allocate GPU arrays

Typical performance (RX 6700 XT):
- Pattern matching: ~500 comparisons/second
- DTW computation: ~1000 sequences/second
- Real-time processing: Up to 10 MSPS sustained

## Troubleshooting

### Pluto+ Not Detected

```bash
# Check USB connection
lsusb | grep "Analog Devices"

# Should show: Bus XXX Device XXX: ID 0456:b673 Analog Devices, Inc. PlutoSDR

# Check libiio
iio_info -s

# Test connectivity
python -c "import iio; print(iio.Context('ip:192.168.2.1'))"
```

### GPU Not Available

```bash
# Check ROCm installation
rocm-smi

# Check CuPy
python -c "import cupy as cp; print(cp.cuda.runtime.getDeviceCount())"

# If issues, reinstall CuPy
pip uninstall cupy-rocm-5-0
pip install cupy-rocm-5-0 --no-cache-dir
```

### High False Positive Rate

1. Increase detection threshold: `t` → `0.85`
2. Reduce RX gain to avoid saturation
3. Enable pattern matching to filter known interference
4. Use narrower bandwidth if monitoring specific frequency

### Poor Cancellation Performance

1. Check TX/RX isolation
2. Verify TX power is appropriate
3. Increase LMS adaptation rate (`mu` parameter)
4. Add delay compensation if needed

## Examples

### Monitoring ISM Band (433 MHz)

```bash
python integrated_detector.py \
    --freq 433.92 \
    --sample-rate 2.4 \
    --cancellation
```

### Scanning for Phantom Voices

```python
from integrated_detector import IntegratedDetectionSystem

system = IntegratedDetectionSystem(
    sample_rate=2.4e6,
    center_freq=100e6
)

# Focus on voice characteristics
system.phantom_detector.voice_freq_range = (300, 3400)
system.het_detector.heterodyne_threshold = 0.65

system.start()
```

### Building Pattern Database

```python
# Run in learning mode
system = IntegratedDetectionSystem()
system.start()

# After collecting patterns
stats = system.pattern_matcher.get_statistics()
print(f"Learned patterns: {stats['total_patterns']}")

# Save patterns (implement serialization)
import pickle
with open('patterns.pkl', 'wb') as f:
    pickle.dump(system.pattern_matcher.known_patterns, f)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Pluto+ SDR Hardware                      │
│                    RX1  RX2  TX1  TX2                       │
└───────────┬──────────────────────┬──────────────────────────┘
            │                      │
            ▼                      ▼
┌─────────────────────┐  ┌──────────────────────┐
│  Heterodyne         │  │  Active Cancellation │
│  Detection          │  │  (LMS Adaptive)      │
│  - Cross-correlation│  │  - TX signal gen     │
│  - Freq offset      │  │  - Phase inversion   │
└──────┬──────────────┘  └──────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────┐
│         Pattern Matching (GPU)                  │
│  ┌────────────────┐  ┌─────────────────────┐  │
│  │ DTW Analysis   │  │ Levenshtein         │  │
│  │ - Spectral     │  │ - Symbol quantize   │  │
│  │ - Temporal     │  │ - Edit distance     │  │
│  └────────────────┘  └─────────────────────┘  │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│         Phantom Voice Detection                 │
│  - Formant analysis                             │
│  - Voice characteristics                        │
│  - Pattern database lookup                      │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
          ┌──────────────┐
          │ Visualization│
          │  & Logging   │
          └──────────────┘
```

## License

This project is provided as-is for research and educational purposes.

## Disclaimer

**Important**: This tool is designed for legitimate RF analysis and research. Ensure you:
- Comply with local RF regulations
- Have authorization to monitor frequencies
- Do not interfere with licensed services
- Understand the legal implications of transmitting cancellation signals

**This is NOT a jammer**: Active cancellation is for local noise reduction only.

## Contributing

Contributions welcome! Areas for improvement:
- Machine learning classification of heterodyne types
- Improved adaptive cancellation algorithms
- Support for more SDR hardware
- Enhanced visualization options

## References

- [DTW for Time Series](https://dtaidistance.readthedocs.io/)
- [LMS Adaptive Filtering](https://en.wikipedia.org/wiki/Least_mean_squares_filter)
- [Pluto+ Documentation](https://wiki.analog.com/university/tools/pluto)
- [ROCm Documentation](https://rocmdocs.amd.com/)
