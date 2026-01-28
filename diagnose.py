#!/usr/bin/env python3
"""
Diagnostic and Setup Script for Heterodyne Detector
Checks dependencies and helps troubleshoot issues
"""

import sys
import subprocess
import importlib

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    print(f"\n{'='*60}")
    print("Python Version Check")
    print(f"{'='*60}")
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required")
        return False
    print("✅ Python version OK")
    return True

def check_module(module_name, package_name=None, install_cmd=None):
    """Check if a Python module is available"""
    if package_name is None:
        package_name = module_name
    
    try:
        mod = importlib.import_module(module_name)
        version = getattr(mod, '__version__', 'unknown')
        print(f"✅ {package_name:20s} - {version}")
        return True
    except ImportError:
        print(f"❌ {package_name:20s} - NOT INSTALLED")
        if install_cmd:
            print(f"   Install: {install_cmd}")
        return False

def check_gpu():
    """Check GPU availability"""
    print(f"\n{'='*60}")
    print("GPU Check")
    print(f"{'='*60}")
    
    # Try CuPy
    try:
        import cupy as cp
        device = cp.cuda.Device(0)
        print(f"✅ CuPy detected")
        print(f"   GPU: {device.compute_capability}")
        
        # Check available memory
        meminfo = device.mem_info
        print(f"   Free memory: {meminfo[0]/1e9:.2f} GB")
        print(f"   Total memory: {meminfo[1]/1e9:.2f} GB")
        return True
    except ImportError:
        print("❌ CuPy not installed")
        print("   For AMD RX 6700 XT, install ROCm version:")
        print("   pip install cupy-rocm-5-0")
        return False
    except Exception as e:
        print(f"⚠️  CuPy installed but GPU not accessible: {e}")
        return False

def check_iio():
    """Check IIO library availability"""
    print(f"\n{'='*60}")
    print("IIO Library Check (for Pluto+ SDR)")
    print(f"{'='*60}")
    
    # Method 1: Check pyadi-iio (recommended)
    try:
        import adi
        print("✅ pyadi-iio (adi) - RECOMMENDED")
        print(f"   Version: {getattr(adi, '__version__', 'unknown')}")
        print("   This is the recommended library for Pluto+")
        return True
    except ImportError:
        print("❌ pyadi-iio not installed")
        print("   Install: pip install pyadi-iio")
    
    # Method 2: Check libiio Python bindings
    try:
        import iio
        print("✅ libiio Python bindings")
        
        # Test for the problematic function
        try:
            # This will fail with older versions
            test = iio._lib.iio_get_backends_count
            print("   libiio version appears compatible")
            return True
        except AttributeError:
            print("⚠️  libiio version mismatch detected!")
            print("   Recommendation: Use pyadi-iio instead")
            print("   pip install pyadi-iio")
            return False
            
    except ImportError:
        print("❌ libiio Python bindings not installed")
    
    return False

def check_pluto_connection():
    """Check if Pluto+ is connected"""
    print(f"\n{'='*60}")
    print("Pluto+ Connection Check")
    print(f"{'='*60}")
    
    # Check USB
    try:
        result = subprocess.run(['lsusb'], capture_output=True, text=True)
        if 'Analog Devices' in result.stdout or '0456:b67' in result.stdout:
            print("✅ Pluto+ detected on USB")
            for line in result.stdout.split('\n'):
                if 'Analog Devices' in line or '0456:b67' in line:
                    print(f"   {line.strip()}")
        else:
            print("❌ Pluto+ not detected on USB")
            print("   Check USB connection")
            return False
    except FileNotFoundError:
        print("⚠️  lsusb not available (are you on Linux?)")
    
    # Try to connect with pyadi-iio
    try:
        import adi
        print("\nAttempting connection...")
        
        # Try USB first
        try:
            sdr = adi.Pluto("usb:1.100.5")
            print("✅ Connected via USB!")
            del sdr
            return True
        except:
            pass
        
        # Try IP
        try:
            sdr = adi.Pluto("ip:192.168.2.1")
            print("✅ Connected via IP (192.168.2.1)!")
            del sdr
            return True
        except:
            pass
        
        print("❌ Could not connect to Pluto+")
        print("   Troubleshooting:")
        print("   1. Check USB cable")
        print("   2. Try: iio_info -u")
        print("   3. Check /var/log/dmesg for USB issues")
        return False
        
    except ImportError:
        print("⚠️  Cannot test connection (pyadi-iio not installed)")
        return False

def check_required_packages():
    """Check all required packages"""
    print(f"\n{'='*60}")
    print("Python Package Check")
    print(f"{'='*60}")
    
    packages = [
        ('numpy', 'numpy', None),
        ('scipy', 'scipy', None),
        ('matplotlib', 'matplotlib', None),
        ('dtaidistance', 'dtaidistance', 'pip install dtaidistance'),
        ('Levenshtein', 'python-Levenshtein', 'pip install python-Levenshtein'),
    ]
    
    all_ok = True
    for module, package, install in packages:
        if not check_module(module, package, install):
            all_ok = False
    
    return all_ok

def suggest_fixes():
    """Suggest fixes for common issues"""
    print(f"\n{'='*60}")
    print("Recommended Actions")
    print(f"{'='*60}")
    
    print("\n1. Install pyadi-iio (REQUIRED):")
    print("   pip install pyadi-iio")
    
    print("\n2. Install other requirements:")
    print("   pip install -r requirements.txt")
    
    print("\n3. For GPU acceleration (AMD RX 6700 XT):")
    print("   # First install ROCm")
    print("   # Then install CuPy:")
    print("   pip install cupy-rocm-5-0")
    
    print("\n4. If Pluto+ not detected:")
    print("   # Check it's connected:")
    print("   lsusb | grep 'Analog Devices'")
    print("   # Test with iio_info:")
    print("   iio_info -u")
    
    print("\n5. Test connection:")
    print("   python -c \"import adi; sdr = adi.Pluto(); print('OK')\"")

def create_test_script():
    """Create a simple test script"""
    test_code = '''#!/usr/bin/env python3
"""Simple test to verify Pluto+ connection"""

import adi
import numpy as np

print("Testing Pluto+ connection...")

# Connect
try:
    sdr = adi.Pluto("usb:1.100.5")
except:
    print("USB failed, trying IP...")
    sdr = adi.Pluto("ip:192.168.2.1")

print(f"Connected!")

# Configure
sdr.sample_rate = int(2.4e6)
sdr.rx_lo = int(100e6)
sdr.rx_rf_bandwidth = int(2.4e6)
sdr.rx_buffer_size = 1024
sdr.rx_enabled_channels = [0, 1]

print(f"Sample rate: {sdr.sample_rate/1e6} MHz")
print(f"Center freq: {sdr.rx_lo/1e6} MHz")

# Receive samples
print("Reading samples...")
samples = sdr.rx()
print(f"Received {len(samples)} channels")
print(f"Channel 0: {len(samples[0])} samples")
print(f"Channel 1: {len(samples[1])} samples")

print("✅ Test successful!")
'''
    
    with open('test_pluto.py', 'w') as f:
        f.write(test_code)
    
    print(f"\n{'='*60}")
    print("Test Script Created")
    print(f"{'='*60}")
    print("Created: test_pluto.py")
    print("Run with: python test_pluto.py")

def main():
    """Main diagnostic routine"""
    print("="*60)
    print("HETERODYNE DETECTOR DIAGNOSTIC TOOL")
    print("="*60)
    
    results = {
        'python': check_python_version(),
        'packages': check_required_packages(),
        'iio': check_iio(),
        'gpu': check_gpu(),
        'pluto': check_pluto_connection()
    }
    
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    
    for key, value in results.items():
        status = "✅" if value else "❌"
        print(f"{status} {key.capitalize()}")
    
    if not results['iio']:
        print("\n⚠️  CRITICAL: pyadi-iio not installed!")
        print("This is required for Pluto+ communication.")
        print("Install with: pip install pyadi-iio")
    
    if not results['pluto']:
        print("\n⚠️  WARNING: Could not connect to Pluto+")
        print("Check USB connection and drivers")
    
    if not results['gpu']:
        print("\n⚠️  INFO: GPU not available")
        print("Pattern matching will use CPU (slower but functional)")
    
    suggest_fixes()
    
    # Offer to create test script
    create = input("\nCreate test_pluto.py test script? (y/n): ")
    if create.lower() == 'y':
        create_test_script()
    
    print("\n" + "="*60)
    print("Diagnostic complete!")
    print("="*60)

if __name__ == "__main__":
    main()
