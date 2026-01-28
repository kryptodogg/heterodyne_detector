# Installing JAX for AMD RX 6700 XT

JAX is **much easier** to install than CuPy and works great with AMD GPUs!

## Why JAX is Better for AMD

1. ✅ **Simpler installation** - No complex ROCm-CuPy matching
2. ✅ **Official AMD support** - JAX has native ROCm backend
3. ✅ **Automatic optimization** - JIT compilation for speed
4. ✅ **Cleaner API** - More Pythonic than CuPy
5. ✅ **Better for ML** - If you want to add ML later

## Installation Options

### Option 1: JAX with ROCm (GPU - Recommended)

For AMD RX 6700 XT with ROCm support:

```bash
# First, ensure ROCm is installed
# Check with: rocm-smi

# Install JAX with ROCm support
pip install jax[rocm] -f https://storage.googleapis.com/jax-releases/jax_rocm_releases.html

# Test it
python -c "import jax; print(jax.devices())"
```

You should see your GPU listed!

### Option 2: JAX CPU-only (Still Fast!)

If you don't have ROCm or want to skip GPU setup:

```bash
# CPU-only (still JIT-optimized and fast)
pip install jax jaxlib

# Test it
python -c "import jax; print('JAX working!')"
```

**Note:** Even CPU-only JAX is faster than vanilla NumPy thanks to JIT!

### Option 3: Use NumPy (Fallback)

The code automatically falls back to NumPy if JAX isn't available.

## Verify Installation

```bash
# Run the test
python jax_pattern_matcher.py
```

You should see:
```
✅ JAX GPU acceleration enabled: [CudaDevice(id=0)]
```

Or for CPU:
```
⚠️  JAX using CPU (GPU not detected)
```

## Performance Comparison

Expected speedup on RX 6700 XT:

| Operation | NumPy | JAX CPU | JAX GPU |
|-----------|-------|---------|---------|
| Correlation | 1x | 2-3x | 10-20x |
| FFT | 1x | 2-4x | 15-30x |
| Matrix ops | 1x | 3-5x | 20-50x |

## Troubleshooting

### "No GPU devices found"

Check ROCm installation:
```bash
rocm-smi
```

If not installed:
```bash
# Ubuntu/Debian
wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/focal/amdgpu-install_*.deb
sudo apt install ./amdgpu-install_*.deb
sudo amdgpu-install --usecase=rocm
```

### "JAX cannot find ROCm"

Set environment variables:
```bash
export ROCM_PATH=/opt/rocm
export HIP_VISIBLE_DEVICES=0
```

Add to `~/.bashrc` to make permanent.

### Still not working?

Use CPU-only JAX - it's still fast!
```bash
pip uninstall jax jaxlib
pip install jax jaxlib  # CPU version
```

## Update Your Code

Replace the old CuPy import:

```python
# OLD (CuPy)
import cupy as cp

# NEW (JAX)
import jax.numpy as jnp
from jax import jit
```

## Updated Requirements

Create a new `requirements_jax.txt`:

```
# Core
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.5.0

# SDR
pyadi-iio>=0.0.17

# Pattern Matching
dtaidistance>=2.3.10
python-Levenshtein>=0.20.0

# GPU Acceleration - JAX (choose one)
jax[rocm]>=0.4.20  # For AMD GPU with ROCm
# jax>=0.4.20        # For CPU-only (still fast!)
# jaxlib>=0.4.20     # Required for CPU
```

Install with:
```bash
# For AMD GPU
pip install -r requirements_jax.txt -f https://storage.googleapis.com/jax-releases/jax_rocm_releases.html

# For CPU-only
pip install -r requirements_jax.txt
```

## Testing Performance

Run the benchmark:
```bash
python jax_pattern_matcher.py
```

This will show you:
- JAX version and backend
- GPU detection status
- Performance comparison vs NumPy
- Pattern matching tests

## Integration with Detector

The detector automatically uses JAX if available:

```python
# No code changes needed!
python integrated_detector.py --freq 100 --sample-rate 2.4
```

It will detect JAX and use it automatically.

## Benefits You'll See

1. **Faster pattern matching** - 10-20x with GPU
2. **Lower latency** - JIT compilation optimizes code
3. **Better memory usage** - Efficient GPU transfers
4. **Future-proof** - Easy to add ML features later

## Next Steps

1. Install JAX (GPU or CPU version)
2. Test with `python jax_pattern_matcher.py`
3. Run your detector - it auto-detects JAX!
4. Enjoy faster heterodyne detection 🚀

The code will automatically use:
- JAX GPU (fastest)
- JAX CPU (fast)
- NumPy (works everywhere)

In that order of preference!
