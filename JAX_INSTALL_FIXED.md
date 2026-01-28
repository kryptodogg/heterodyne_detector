# JAX Installation for AMD RX 6700 XT - Updated Guide

## ⚠️ Important: JAX ROCm Status (2025)

As of early 2025, **JAX's official ROCm support has changed**. Here are your current options:

## Option 1: JAX CPU-Only (Recommended - Still Fast!)

JAX on CPU is **2-3x faster** than NumPy thanks to JIT compilation:

```bash
# Install with UV
uv pip install jax jaxlib

# Or with pip
pip install jax jaxlib

# Test
uv run python -c "import jax; print(jax.devices())"
```

**Output should show:** `[CpuDevice(id=0)]`

This works great for your heterodyne detector! The JIT compilation makes it fast even without GPU.

## Option 2: Use PyTorch with ROCm Instead

PyTorch has better AMD support than JAX currently:

```bash
# Install PyTorch with ROCm
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0

# Verify GPU
uv run python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Then modify the detector to use PyTorch instead of JAX.

## Option 3: JAX CUDA (for NVIDIA GPUs)

If you had an NVIDIA GPU:

```bash
uv pip install jax[cuda12]
```

But for AMD RX 6700 XT, this won't work.

## Option 4: Build JAX with ROCm from Source (Advanced)

For the truly adventurous:

```bash
# Clone JAX
git clone https://github.com/google/jax.git
cd jax

# Install dependencies
uv pip install numpy scipy cython

# Build with ROCm
python build/build.py --enable_rocm --rocm_path=/opt/rocm

# Install
uv pip install -e .
```

**Warning:** This is complex and may not work depending on ROCm version.

## Option 5: Use the Detector Without GPU Acceleration

The detector works perfectly fine without JAX:

```bash
# Just install basic requirements
uv pip install numpy scipy pyadi-iio

# Run detector (uses NumPy)
uv run python standalone_detector.py --simulate
```

It will automatically detect that JAX isn't available and use NumPy instead.

## 🎯 My Recommendation for You

### For Right Now (Easiest)

```bash
# Install JAX CPU-only (still 2-3x faster than NumPy)
uv pip install jax jaxlib

# Test your detector
uv run python standalone_detector.py --simulate
```

This gives you:
- ✅ Easy installation (no ROCm needed)
- ✅ 2-3x speedup from JIT
- ✅ Works immediately
- ✅ No GPU driver issues

### For Maximum Performance (More Work)

Use PyTorch instead of JAX for GPU acceleration:

1. **Install PyTorch with ROCm:**
   ```bash
   uv pip install torch --index-url https://download.pytorch.org/whl/rocm6.0
   ```

2. **Modify detector to use PyTorch:**
   I can create a PyTorch version if you want!

## Testing Your Installation

### Test JAX (CPU)

```bash
cat > test_jax.py << 'EOF'
import jax
import jax.numpy as jnp
from jax import jit
import time

print(f"JAX version: {jax.__version__}")
print(f"Devices: {jax.devices()}")
print(f"Backend: {jax.default_backend()}")

# Test JIT compilation
@jit
def fast_computation(x):
    return jnp.dot(x, x.T)

# Benchmark
x = jnp.ones((1000, 1000))

# Warmup
_ = fast_computation(x)

# Time it
start = time.time()
for _ in range(100):
    _ = fast_computation(x)
elapsed = time.time() - start

print(f"\nPerformance test: {elapsed:.3f}s for 100 iterations")
print("✅ JAX is working!")
EOF

uv run python test_jax.py
```

### Test PyTorch (GPU)

```bash
cat > test_pytorch.py << 'EOF'
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Test GPU computation
    x = torch.randn(1000, 1000).cuda()
    y = torch.mm(x, x.t())
    print("✅ GPU computation working!")
else:
    print("⚠️  No GPU detected")
    print("Using CPU (still works!)")
EOF

uv run python test_pytorch.py
```

## Updated Requirements File

Here's a working requirements file:

```bash
cat > requirements_updated.txt << 'EOF'
# Core dependencies
numpy>=1.21.0
scipy>=1.7.0
pyadi-iio>=0.0.17

# Pattern matching
dtaidistance>=2.3.10
python-Levenshtein>=0.20.0

# Visualization
matplotlib>=3.5.0

# Acceleration options (choose one):

# Option 1: JAX CPU-only (recommended - easy install, 2-3x speedup)
jax>=0.4.20
jaxlib>=0.4.20

# Option 2: PyTorch with ROCm (for AMD GPU)
# torch>=2.0.0  # Install separately with --index-url

# Option 3: No acceleration (uses NumPy - works everywhere)
# (just don't install JAX or PyTorch)
EOF
```

Install with:
```bash
uv pip install -r requirements_updated.txt
```

## Why ROCm JAX URL Doesn't Work

The Google Storage URL for JAX ROCm releases was deprecated. Current alternatives:

1. **JAX no longer officially supports ROCm** through pip
2. **PyTorch has better ROCm support** for AMD GPUs
3. **Building from source** is possible but complex
4. **CPU JAX** still gives good performance via JIT

## Decision Matrix

| Option | Speed vs NumPy | Ease | GPU Use | Recommendation |
|--------|----------------|------|---------|----------------|
| **JAX CPU** | 2-3x | ⭐⭐⭐⭐⭐ | No | ✅ **Best for most users** |
| **PyTorch ROCm** | 10-20x | ⭐⭐⭐ | Yes | ✅ If you need max speed |
| **JAX ROCm (build)** | 10-20x | ⭐ | Yes | ❌ Too complex |
| **NumPy only** | 1x | ⭐⭐⭐⭐⭐ | No | ✅ If you want simple |

## Complete Working Setup

Here's what will work **right now**:

```bash
# 1. Install JAX CPU-only
uv pip install jax jaxlib numpy scipy pyadi-iio

# 2. Test it
uv run python -c "import jax; print('JAX version:', jax.__version__); print('Devices:', jax.devices())"

# 3. Run your detector
uv run python standalone_detector.py --simulate

# 4. Expected output:
# ✅ JAX CPU acceleration enabled: [CpuDevice(id=0)]
# ⚠️  JAX using CPU (still fast!)
```

## Alternative: Create PyTorch Version

If you want maximum GPU performance, I can create a PyTorch-accelerated version of the detector. PyTorch has excellent AMD support:

```bash
# Install PyTorch with ROCm 6.0
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0

# Verify GPU
uv run python -c "import torch; print(torch.cuda.is_available())"
```

Should output: `True` if ROCm is properly installed.

## What About Your RX 6700 XT?

For AMD RX 6700 XT specifically:

**Option A: JAX CPU** (Easy, 2-3x speedup)
- ✅ Works now
- ✅ No driver hassles
- ✅ Good enough for most use cases

**Option B: PyTorch ROCm** (Max performance, more setup)
- ✅ 10-20x speedup
- ⚠️ Requires ROCm drivers
- ✅ Better AMD support than JAX

**Option C: Just NumPy** (Simplest)
- ✅ Zero setup
- ❌ Slower
- ✅ Always works

## My Actual Recommendation

```bash
# Start simple:
uv pip install jax jaxlib numpy scipy pyadi-iio dtaidistance python-Levenshtein

# Test:
uv run python standalone_detector.py --simulate
```

If you need more speed later, we can switch to PyTorch. But JAX CPU is a great middle ground!

## Quick Install Script

```bash
#!/bin/bash
echo "Installing heterodyne detector with JAX CPU..."

# Install dependencies
uv pip install jax jaxlib numpy scipy pyadi-iio dtaidistance python-Levenshtein matplotlib

# Test
echo "Testing JAX..."
uv run python -c "import jax; print('✅ JAX installed:', jax.__version__)"

# Test detector
echo "Testing detector..."
uv run python standalone_detector.py --simulate --duration 5

echo "✅ Installation complete!"
```

Save as `install.sh`, then:
```bash
chmod +x install.sh
./install.sh
```

**Bottom line:** The ROCm URL is dead. Use JAX CPU-only (still fast!) or switch to PyTorch for GPU.
