# AGENTS.md - Development Guidelines for Heterodyne Detector

## Build/Test Commands

### Dependencies Installation
```bash
# Step 1: Install ROCm (AMD GPU) or skip for CPU-only
wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/focal/amdgpu-install_*.deb
sudo apt install ./amdgpu-install_*.deb
sudo amdgpu-install --usecase=rocm

# Step 2: Install PyTorch with ROCm (CRITICAL - install FIRST)
# Using Conda environment:
conda activate your-env-name
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.1

# Step 3: Install remaining dependencies
pip install -r requirements.txt

# Alternative CPU-only installation
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

### Running Tests
```bash
# Run all tests
python -m pytest tests/ -v

# Run single test file
python -m pytest tests/test_radar_app.py -v

# Run specific test method
python -m pytest tests/test_radar_app.py::TestRadarApp::test_init_buffers -v

# Run tests with GPU verification
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}')" && python -m pytest tests/ -v
```

### Application Execution
```bash
# Simulation mode (no hardware required)
python main.py --simulate --duration 10

# Real hardware mode
python main.py --freq 2400  # 2.4 GHz

# Custom configuration
python main.py --config radar.yaml

# Diagnostic tools
python diagnose.py  # Check system dependencies
./fix_libiio.sh     # Fix SDR interface issues (interactive)
```

### Validation Commands
```bash
# Verify GPU acceleration
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# Test SDR connection
python -c "import adi; sdr = adi.Pluto(); print('Pluto+ Connected!')"  # Requires hardware

# Validate core imports
python -c "import dash; import h5py; import torch; print('All dependencies OK')"
```

## Code Style Guidelines

### Architecture Principles (Torch-First)
1. **Zero-Copy GPU Pipeline**: All data enters as PyTorch tensors on GPU. Use `torch.from_numpy().pin_memory().cuda(non_blocking=True)` for DMA transfers
2. **Module Composition**: Each processor (SpatialNoiseCanceller, HeterodyneDetector, etc.) is independent and composable
3. **Geometry-Driven**: Radar geometry (antenna positions, baselines) is a first-class citizen driving beamforming

### Import Organization
```python
# Standard library imports first
import argparse
import sys
import time
from pathlib import Path
from datetime import datetime

# Third-party scientific computing
import numpy as np
import torch
import torch.nn as nn

# Project imports (grouped by functionality)
from config import (
    RADAR_GEOMETRY,
    NOISE_CANCELLATION,
    GPU_CONFIG,
    get_torch_geometry
)
from noise_canceller import SpatialNoiseCanceller
from heterodyne_detector import HeterodyneDetector
```

### Tensor Management
- **Device Consistency**: All tensors must be on the same device (CPU/GPU). Check with `tensor.device.type`
- **Pre-allocation**: Reuse buffers to avoid fragmentation. Pre-allocate in `__init__()` methods
- **Complex Tensors**: Use `dtype=torch.complex64` for IQ signals, `torch.float32` for real-valued data
- **Memory Monitoring**: Track GPU usage with `torch.cuda.memory_allocated()` and `torch.cuda.memory_reserved()`

### Naming Conventions
- **Classes**: PascalCase (e.g., `SpatialNoiseCanceller`, `RadarApp`)
- **Functions/Methods**: snake_case (e.g., `process_buffer()`, `compute_steering_vector()`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `RADAR_GEOMETRY`, `GPU_CONFIG`)
- **Private Methods**: Prefix with underscore (e.g., `_mvdr_beamform()`, `_adaptive_lms()`)
- **Tensor Variables**: Descriptive names ending with `_tensor` or `_buffer` (e.g., `rx1_buffer`, `steering_vectors`)

### Type Hints
```python
from typing import Dict, List, Tuple, Optional
import torch

def process_buffer(
    self, 
    rx1: torch.Tensor, 
    rx2: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """Process dual-channel radar signals."""
    pass

def compute_steering_vector(
    self, 
    angles: torch.Tensor
) -> torch.Tensor:
    """Return steering vectors for given angles."""
    pass
```

### Error Handling
- **Graceful Degradation**: CPU fallback when GPU unavailable
- **Hardware Validation**: Check SDR connection before use
- **Memory Safety**: Validate tensor shapes and devices before operations
- **Configuration Validation**: Verify geometry consistency and parameter ranges

```python
# Example error handling pattern
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
except Exception as e:
    print(f"Warning: GPU initialization failed: {e}")
    device = torch.device('cpu')

if rx1.device != rx2.device:
    raise ValueError("RX1 and RX2 must be on the same device")
```

### Configuration Management
- **Centralized Config**: All parameters in `config.py` with clear sections
- **Geometry Validation**: Ensure antenna positions are physically realistic
- **GPU Memory Tuning**: Adjust buffer sizes based on available VRAM
- **Algorithm Thresholds**: Make detection parameters configurable

### Documentation Standards
- **Module Headers**: Include purpose, key algorithms, and usage examples
- **Method Docstrings**: Document inputs, outputs, and algorithmic approach
- **Configuration Comments**: Explain physical meaning of parameters
- **GPU Optimization Notes**: Document zero-copy patterns and memory considerations

### Testing Patterns
- **Unit Tests**: Test each module independently with synthetic data
- **Integration Tests**: Verify end-to-end pipeline with simulation
- **GPU Tests**: Ensure tensor device consistency and memory management
- **Hardware Tests**: Validate SDR interface (only when hardware available)

### Performance Guidelines
- **Processing Latency**: Target <1ms per 10k-sample buffer on RX 6700 XT
- **Memory Footprint**: Aim for 2-4 GB total GPU usage
- **Visualization Refresh**: 20-60 Hz depending on plot complexity
- **SNR Improvement**: Expect 6-12 dB from combined MVDR + LMS processing

### Safety & Regulatory
- **RF Power Limits**: Respect ISM band regulations via TX gain settings
- **Data Logging**: All detections logged to HDF5 with timestamps
- **Active Cancellation**: Disabled by default; enable explicitly
- **Hardware Protection**: Validate parameters before SDR configuration

## File Organization Pattern

```
main.py                    # RadarApp orchestrator
├─ config.py              # All configuration (geometry, GPU, algorithms)
├─ sdr_interface.py       # PlutoSDR hardware interface
├─ heterodyne_detector.py # IQ detection + heterodyning
├─ noise_canceller.py     # MVDR + LMS beamforming
├─ range_doppler.py       # Doppler velocity processing
├─ ppi_processor.py       # Polar position indicator
├─ tracker.py             # Kalman filter tracking
├─ pattern_matcher.py     # ML-based signal classification
├─ visualizer.py          # Real-time Dash dashboard
├─ audio_processor.py     # MFCC feature extraction
└─ data_manager.py        # HDF5 library management
```

Remember: This is a GPU-accelerated radar system using Torch-first architecture. Every tensor operation should stay on GPU when possible, and geometry awareness is critical for beamforming accuracy.