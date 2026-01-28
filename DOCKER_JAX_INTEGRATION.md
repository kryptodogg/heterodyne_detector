# Using Docker JAX with Python/Conda/UV

There are several ways to make Docker JAX available to your Python environment. Here are the best options:

## Option 1: Use JAX Docker Container Directly (Recommended)

Run your Python code **inside** the JAX Docker container:

### With Docker Run

```bash
# Pull official JAX Docker image
docker pull gcr.io/tensorflow-testing/nosla-cuda12.3-cudnn8.9-ubuntu22.04-manylinux2014-multipython

# Or for ROCm (AMD):
docker pull rocm/pytorch:latest

# Run your script inside the container
docker run --rm \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    -v $(pwd):/workspace \
    -w /workspace \
    rocm/pytorch:latest \
    bash -c "pip install jax[rocm] scipy pyadi-iio && python standalone_detector.py --simulate"
```

### With Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  jax-detector:
    image: rocm/pytorch:latest
    devices:
      - /dev/kfd
      - /dev/dri
    group_add:
      - video
    volumes:
      - ./:/workspace
    working_dir: /workspace
    environment:
      - HSA_OVERRIDE_GFX_VERSION=10.3.0  # For RX 6700 XT
      - ROCM_PATH=/opt/rocm
    command: bash -c "pip install jax[rocm] scipy numpy pyadi-iio && python standalone_detector.py --simulate"
```

Run with:
```bash
docker-compose up
```

## Option 2: Bind Mount Docker JAX to Host (Advanced)

Make Docker's JAX installation visible to host Python:

### Step 1: Create Docker Container with JAX

```bash
# Create Dockerfile
cat > Dockerfile.jax << 'EOF'
FROM rocm/pytorch:latest

# Install JAX with ROCm
RUN pip install --prefix=/jax-install \
    jax[rocm] \
    -f https://storage.googleapis.com/jax-releases/jax_rocm_releases.html

# Install other dependencies
RUN pip install --prefix=/jax-install \
    numpy scipy
EOF

# Build
docker build -t jax-install -f Dockerfile.jax .

# Run and copy out the installation
docker run --rm -v $(pwd)/jax_libs:/output jax-install \
    cp -r /jax-install /output/
```

### Step 2: Add to Python Path

```bash
# Add to your environment
export PYTHONPATH="$(pwd)/jax_libs/jax-install/lib/python3.10/site-packages:$PYTHONPATH"
export LD_LIBRARY_PATH="$(pwd)/jax_libs/jax-install/lib:$LD_LIBRARY_PATH"

# Test
python -c "import jax; print(jax.devices())"
```

**Problem:** This is fragile and doesn't always work due to ABI mismatches.

## Option 3: Use UV with Docker Backend (Cleanest)

UV can use Docker containers as backends:

### Create `.python-version-docker`

```bash
# Tell UV to use Docker for this project
cat > .python-version-docker << 'EOF'
docker://rocm/pytorch:latest
EOF
```

### Configure UV

```bash
# In your project directory
cat > uv.toml << 'EOF'
[tool.uv]
runtime = "docker"
docker-image = "rocm/pytorch:latest"
docker-devices = ["/dev/kfd", "/dev/dri"]
docker-groups = ["video"]
EOF
```

### Run with UV

```bash
# UV will automatically use Docker
uv run python standalone_detector.py --simulate
```

**Note:** This is experimental in UV. Check `uv --version` for Docker support.

## Option 4: Conda with Docker (Hybrid)

Use Conda for most packages, Docker JAX via wrapper:

### Create Wrapper Script

```bash
# Create jax_docker_wrapper.py
cat > jax_docker_wrapper.py << 'PYTHON'
#!/usr/bin/env python3
"""
JAX Docker Wrapper - Transparently uses JAX from Docker
"""
import subprocess
import sys
import pickle
import base64

def run_in_docker(code, *args):
    """Run Python code in Docker with JAX"""
    # Serialize arguments
    args_b64 = base64.b64encode(pickle.dumps(args)).decode()
    
    # Prepare code
    full_code = f"""
import pickle
import base64
args = pickle.loads(base64.b64decode('{args_b64}'.encode()))

# User code
{code}
"""
    
    # Run in Docker
    result = subprocess.run([
        'docker', 'run', '--rm',
        '--device=/dev/kfd',
        '--device=/dev/dri',
        '--group-add', 'video',
        'rocm/pytorch:latest',
        'python', '-c', full_code
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Docker execution failed: {result.stderr}")
    
    return result.stdout

# Example usage
if __name__ == "__main__":
    code = """
import jax
import jax.numpy as jnp

x = jnp.array([1, 2, 3])
y = jnp.array([4, 5, 6])
result = jnp.dot(x, y)
print(f"Result: {result}")
"""
    print(run_in_docker(code))
PYTHON

chmod +x jax_docker_wrapper.py

# Use it
python jax_docker_wrapper.py
```

## Option 5: Install Native JAX with ROCm (Best Long-term)

Forget Docker, install JAX natively:

### Install ROCm First

```bash
# Ubuntu/Debian
wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/focal/amdgpu-install_*.deb
sudo apt install ./amdgpu-install_*.deb
sudo amdgpu-install --usecase=rocm

# Verify
rocm-smi
```

### Install JAX with ROCm

```bash
# Set environment
export ROCM_PATH=/opt/rocm
export HIP_VISIBLE_DEVICES=0

# Install JAX
pip install jax[rocm] -f https://storage.googleapis.com/jax-releases/jax_rocm_releases.html

# Or with UV:
uv pip install jax[rocm] -f https://storage.googleapis.com/jax-releases/jax_rocm_releases.html

# Or with Conda:
conda install -c conda-forge jaxlib
pip install jax[rocm] -f https://storage.googleapis.com/jax-releases/jax_rocm_releases.html

# Test
python -c "import jax; print(jax.devices())"
```

**This is the cleanest long-term solution!**

## Comparison Table

| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| **Run in Docker** | Easy, isolated, guaranteed to work | Slower I/O, more setup | Quick testing |
| **Bind mount** | Uses host Python | Fragile, ABI issues | Not recommended |
| **UV Docker** | Clean integration | Experimental, limited support | Future use |
| **Conda+Docker** | Hybrid approach | Complex | Mixed environments |
| **Native ROCm** | Fastest, cleanest | Requires ROCm install | Production use |

## Recommended Approach for Your Setup

### For Quick Testing (Now)
```bash
# Use Docker directly
docker run --rm \
    --device=/dev/kfd --device=/dev/dri --group-add video \
    -v $(pwd):/work -w /work \
    rocm/pytorch:latest \
    bash -c "pip install jax[rocm] scipy numpy && python standalone_detector.py --simulate"
```

### For Development (Best)
```bash
# Install ROCm and JAX natively
sudo amdgpu-install --usecase=rocm
uv pip install jax[rocm] -f https://storage.googleapis.com/jax-releases/jax_rocm_releases.html
uv run python standalone_detector.py --simulate
```

## UV-Specific Solutions

Since you're using `uv`, here are UV-specific approaches:

### Method 1: UV with System Python + Docker JAX

```bash
# Install most packages with UV
uv pip install numpy scipy pyadi-iio

# Create alias for JAX scripts
alias uv-jax='docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video -v $(pwd):/work -w /work rocm/pytorch:latest'

# Use it
uv-jax python standalone_detector.py --simulate
```

### Method 2: UV Virtual Environment in Docker

```bash
# Create Dockerfile with UV
cat > Dockerfile.uv-jax << 'EOF'
FROM rocm/pytorch:latest

# Install UV
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Setup project
WORKDIR /project
COPY requirements_jax.txt .
RUN uv venv && uv pip install -r requirements_jax.txt

ENTRYPOINT ["uv", "run"]
EOF

# Build
docker build -t uv-jax -f Dockerfile.uv-jax .

# Use
docker run --rm \
    --device=/dev/kfd --device=/dev/dri --group-add video \
    -v $(pwd):/project -w /project \
    uv-jax python standalone_detector.py --simulate
```

### Method 3: UV Tool with JAX Container

```bash
# Create UV tool wrapper
cat > pyproject.toml << 'EOF'
[project]
name = "heterodyne-detector"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.21.0",
    "scipy>=1.7.0",
    "pyadi-iio>=0.0.17",
]

[tool.uv]
dev-dependencies = [
    "jax[rocm]>=0.4.20",
]

[tool.uv.sources]
jax = { url = "https://storage.googleapis.com/jax-releases/jax_rocm_releases.html" }
EOF

# Install
uv sync

# Run
uv run python standalone_detector.py --simulate
```

## Troubleshooting Docker + Host Integration

### Issue: GPU Not Visible in Docker

```bash
# Verify host can see GPU
rocm-smi

# Check Docker can see devices
docker run --rm \
    --device=/dev/kfd --device=/dev/dri \
    rocm/pytorch:latest \
    rocm-smi

# If fails, check permissions
ls -la /dev/kfd /dev/dri/
groups  # Should include 'video' or 'render'
```

### Issue: ABI Mismatch with Bind Mounts

**Solution:** Don't bind mount. Use native install or run entirely in Docker.

### Issue: UV Doesn't See Docker JAX

```bash
# UV uses host Python by default
# Either:
# 1. Install JAX on host
uv pip install jax[rocm] -f https://storage.googleapis.com/jax-releases/jax_rocm_releases.html

# 2. Use Docker entirely
docker run --rm ... uv-jax-image ...
```

## Complete Working Example

Here's a complete setup that works with UV:

```bash
# 1. Create project structure
mkdir heterodyne-detector
cd heterodyne-detector

# 2. Copy your standalone_detector.py here

# 3. Create pyproject.toml
cat > pyproject.toml << 'EOF'
[project]
name = "heterodyne-detector"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.21.0",
    "scipy>=1.7.0",
    "pyadi-iio>=0.0.17",
]

[project.optional-dependencies]
gpu = [
    "jax[rocm]>=0.4.20",
]
EOF

# 4. Install (choose one):

# Option A - No GPU (CPU only, still fast)
uv sync
uv run python standalone_detector.py --simulate

# Option B - With GPU (native install)
sudo amdgpu-install --usecase=rocm  # One-time setup
uv sync --extra gpu --index-url https://storage.googleapis.com/jax-releases/jax_rocm_releases.html
uv run python standalone_detector.py --simulate

# Option C - With GPU (Docker)
docker run --rm \
    --device=/dev/kfd --device=/dev/dri --group-add video \
    -v $(pwd):/work -w /work \
    rocm/pytorch:latest \
    bash -c "pip install uv && uv sync && uv run python standalone_detector.py --simulate"
```

## My Recommendation

**Install JAX natively with UV:**

```bash
# One-time ROCm setup
sudo amdgpu-install --usecase=rocm

# Then use UV normally
uv pip install jax[rocm] -f https://storage.googleapis.com/jax-releases/jax_rocm_releases.html

# Test
uv run python -c "import jax; print(jax.devices())"

# Use your detector
uv run python standalone_detector.py --simulate
```

This gives you:
- ✅ Native performance
- ✅ Works with UV
- ✅ No Docker complexity
- ✅ Easy to debug

Docker is great for deployment, but native install is better for development!
