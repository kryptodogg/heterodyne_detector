#!/usr/bin/env python3
"""
Environment Check for Heterodyne Detector Test Suite
Verifies dependencies and provides installation guidance
"""

import sys
import importlib.util
from pathlib import Path

def check_module(module_name, package_name=None):
    """Check if a module is installed"""
    package_name = package_name or module_name
    spec = importlib.util.find_spec(module_name)
    if spec is not None:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {package_name:20s} {version}")
            return True
        except Exception as e:
            print(f"⚠️  {package_name:20s} Found but import failed: {e}")
            return False
    else:
        print(f"❌ {package_name:20s} NOT INSTALLED")
        return False

def main():
    print("="*60)
    print("Heterodyne Detector - Test Environment Check")
    print("="*60)
    print()

    # Critical dependencies
    print("📦 Critical Dependencies:")
    print("-" * 60)
    torch_ok = check_module('torch', 'PyTorch')
    numpy_ok = check_module('numpy', 'NumPy')
    pytest_ok = check_module('pytest', 'pytest')

    print()
    print("📦 Core Application Dependencies:")
    print("-" * 60)
    scipy_ok = check_module('scipy', 'SciPy')
    h5py_ok = check_module('h5py', 'h5py')
    dash_ok = check_module('dash', 'Dash')
    plotly_ok = check_module('plotly', 'Plotly')

    print()
    print("📦 Optional Dependencies:")
    print("-" * 60)
    adi_ok = check_module('adi', 'pyadi-iio')

    print()
    print("="*60)

    # Check GPU availability if torch is installed
    if torch_ok:
        try:
            import torch
            gpu_available = torch.cuda.is_available()
            if gpu_available:
                device_name = torch.cuda.get_device_name(0)
                print(f"🎮 GPU Status: ✅ Available ({device_name})")
                print(f"   CUDA Version: {torch.version.cuda}")
                print(f"   Device Count: {torch.cuda.device_count()}")
            else:
                print("🎮 GPU Status: ⚠️  Not available (CPU mode only)")
        except Exception as e:
            print(f"🎮 GPU Status: ❌ Error checking: {e}")
    else:
        print("🎮 GPU Status: ❌ PyTorch not installed")

    print("="*60)
    print()

    # Check test files
    print("🧪 Test Files:")
    print("-" * 60)
    test_dir = Path(__file__).parent / 'tests'
    if test_dir.exists():
        test_files = sorted(test_dir.glob('test_*.py'))
        for test_file in test_files:
            print(f"   ✅ {test_file.name}")
        print(f"\n   Total: {len(test_files)} test files")
    else:
        print("   ❌ tests/ directory not found")

    print("="*60)
    print()

    # Provide recommendations
    print("📋 Installation Status:")
    print("-" * 60)

    all_critical = all([torch_ok, numpy_ok, pytest_ok])
    all_core = all([scipy_ok, h5py_ok, dash_ok, plotly_ok])

    if all_critical and all_core:
        print("✅ All critical dependencies installed")
        print("✅ Ready to run tests!")
        print()
        print("Run tests with:")
        print("   python -m pytest tests/ -v")
        return 0
    else:
        print("❌ Missing dependencies detected")
        print()
        print("📝 Installation Instructions:")
        print("-" * 60)

        if not torch_ok:
            print("\n1. Install PyTorch with ROCm (AMD GPU):")
            print("   pip install torch torchvision torchaudio \\")
            print("     --index-url https://download.pytorch.org/whl/rocm6.0")
            print()
            print("   OR for CPU-only:")
            print("   pip install torch torchvision torchaudio")

        if not all([numpy_ok, scipy_ok, h5py_ok, dash_ok, plotly_ok, pytest_ok]):
            print("\n2. Install remaining dependencies:")
            print("   pip install -r requirements.txt")

        print()
        print("3. Verify installation:")
        print("   python check_test_environment.py")

        return 1

if __name__ == '__main__':
    sys.exit(main())
