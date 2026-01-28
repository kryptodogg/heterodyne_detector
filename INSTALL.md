# Quick Start Installation Guide

## Fix for the `iio_get_backends_count` Error

The error you encountered is due to a version mismatch between `pylibiio` and your system's `libiio` library. Here's how to fix it:

## Solution: Use pyadi-iio (Recommended)

The **pyadi-iio** library is more stable and easier to install. It's the official Python support for Analog Devices SDRs.

### Step 1: Uninstall conflicting packages

```bash
pip uninstall pylibiio iio -y
```

### Step 2: Install pyadi-iio

```bash
pip install pyadi-iio
```

### Step 3: Install other requirements

```bash
pip install numpy scipy matplotlib dtaidistance python-Levenshtein
```

### Step 4: Test the installation

```bash
python diagnose.py
```

This will check all dependencies and test your Pluto+ connection.

## Full Installation Steps

### 1. Create virtual environment (recommended)

```bash
cd "Herterodyning Detector"
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
# Core requirements
pip install pyadi-iio numpy scipy matplotlib

# Pattern matching (for DTW/Levenshtein)
pip install dtaidistance python-Levenshtein

# Optional: GPU acceleration for AMD RX 6700 XT
# First ensure ROCm is installed, then:
# pip install cupy-rocm-5-0
```

### 3. Verify Pluto+ connection

```bash
# Check USB connection
lsusb | grep "Analog Devices"

# Should show something like:
# Bus 001 Device 010: ID 0456:b673 Analog Devices, Inc. PlutoSDR (ADALM-PLUTO)

# Run diagnostic
python diagnose.py
```

### 4. Test basic connection

```bash
python -c "import adi; sdr = adi.Pluto(); print('Connected!')"
```

### 5. Run the detector

```bash
python integrated_detector.py --freq 100 --sample-rate 2.4
```

## Troubleshooting

### "Could not find Pluto"

1. **Check USB cable** - Use a good quality USB cable
2. **Check USB port** - Try different USB ports
3. **Driver issues** - On Linux, check dmesg: `dmesg | tail -20`
4. **Network mode** - Try: `python integrated_detector.py` (it will auto-detect)

### "No module named 'adi'"

```bash
pip install pyadi-iio
```

### "GPU not available" warning

This is OK! The system will work fine on CPU. For GPU acceleration:

1. Install ROCm: https://rocmdocs.amd.com/
2. Install CuPy: `pip install cupy-rocm-5-0`

### Still having issues?

Run the diagnostic tool:

```bash
python diagnose.py
```

It will:
- Check all dependencies
- Test Pluto+ connection
- Suggest specific fixes
- Create a simple test script

## Alternative: USB vs Network Mode

### USB Mode (Default)
```python
sdr = adi.Pluto("usb:1.100.5")
```

### Network Mode
```python
sdr = adi.Pluto("ip:192.168.2.1")
```

The code tries both automatically!

## Verify Installation

After installation, you should be able to run:

```bash
python integrated_detector.py --freq 100 --sample-rate 2.4
```

You should see:
```
Initializing Integrated Heterodyne Detection System...
Using pyadi-iio for Pluto+ connection
Connecting to Pluto+ SDR...
Connected to Pluto+ SDR
Sample Rate: 2.40 MHz
Center Freq: 100.00 MHz
RX Gain: 50 dB
```

## What Changed

The original code used the low-level `iio` library which has version compatibility issues. The updated code now uses:

1. **pyadi-iio (adi)** - Higher-level, more stable
2. **Automatic fallback** - Tries multiple connection methods
3. **Better error messages** - Clearer troubleshooting info

## Files Updated

- `heterodyne_detector.py` - Now uses pyadi-iio
- `integrated_detector.py` - Imports updated detector
- `requirements.txt` - Fixed dependencies
- `diagnose.py` - NEW: Diagnostic tool

## Next Steps

1. Run `python diagnose.py` to verify everything
2. Start with basic detection: `python integrated_detector.py`
3. Experiment with different frequencies and settings
4. Enable GPU acceleration once comfortable

## Need Help?

The diagnostic script (`diagnose.py`) provides detailed troubleshooting information and will help identify exactly what's wrong with your setup.
