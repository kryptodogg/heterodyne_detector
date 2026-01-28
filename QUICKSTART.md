# Quick Start - Multiple Options

You have **3 options** to get this working:

## Option 1: Fix libiio (Recommended for Hardware)

Run the automated fix script:

```bash
chmod +x fix_libiio.sh
./fix_libiio.sh
```

This will:
- Build libiio v0.25 from source
- Install matching Python bindings
- Set up udev rules for Pluto+
- Test the installation

After this, run:
```bash
python integrated_detector.py --freq 100 --sample-rate 2.4
```

## Option 2: Test with Simulation (No Hardware Needed)

Test the system without any hardware using the alternative detector:

```bash
python alternative_detector.py --simulate --duration 30
```

This generates synthetic heterodyne signals and phantom voice artifacts for testing. Perfect for:
- Testing the algorithms
- Understanding the detection process
- Development without hardware

## Option 3: Manual libiio Fix

If the automated script fails:

```bash
# 1. Install dependencies
sudo apt install -y build-essential cmake git libxml2-dev bison flex \
    libcdk5-dev libusb-1.0-0-dev libserialport-dev libavahi-client-dev \
    libaio-dev python3-dev

# 2. Build libiio
cd /tmp
git clone https://github.com/analogdevicesinc/libiio.git
cd libiio
git checkout v0.25
mkdir build && cd build
cmake .. -DPYTHON_BINDINGS=ON
make -j$(nproc)
sudo make install
sudo ldconfig

# 3. Install Python bindings from source
cd /tmp/libiio/bindings/python
python setup.py install

# 4. Install pyadi-iio
pip install pyadi-iio

# 5. Test
python -c "import adi; print('Success!')"
```

## Understanding the Error

The error `undefined symbol: iio_get_backends_count` means:
- Your system has an **old version** of libiio (< 0.24)
- Python bindings expect a **newer version** (≥ 0.24)

**Why this happens:**
- Ubuntu/Debian ship with older libiio versions
- Python packages expect newer versions
- Version mismatch = missing symbols

## Quick Test Commands

After fixing libiio:

```bash
# Test 1: Check libiio is installed
iio_info -u

# Test 2: Test Python import
python -c "import iio; import adi; print('✅ OK')"

# Test 3: Test Pluto connection
python -c "import adi; sdr = adi.Pluto(); print('✅ Connected')"

# Test 4: Run detector
python alternative_detector.py --freq 100 --sample-rate 2.4
```

## If You Still Have Issues

### Check libiio version:

```bash
# Find your libiio library
find /usr -name "libiio.so*" 2>/dev/null

# Check what version is installed
dpkg -l | grep libiio
```

### Verify USB connection:

```bash
# Check if Pluto+ is detected
lsusb | grep "Analog Devices"

# Should show:
# Bus XXX Device XXX: ID 0456:b673 Analog Devices, Inc.
```

### Check Python environment:

```bash
# Make sure you're using the right Python
which python
python --version

# Check installed packages
pip list | grep -E "adi|iio|pyadi"
```

## Files Overview

| File | Purpose | When to Use |
|------|---------|-------------|
| `fix_libiio.sh` | Automated fix for libiio | Hardware setup |
| `alternative_detector.py` | Simulation mode | Testing without hardware |
| `integrated_detector.py` | Full system | After libiio is fixed |
| `diagnose.py` | Check your setup | Troubleshooting |

## Expected Output (Simulation Mode)

```
$ python alternative_detector.py --simulate --duration 10

============================================================
Alternative Heterodyne Detector
============================================================
============================================================
SIMULATION MODE
============================================================
Using synthetic data for testing/development
To use real hardware, set simulate=False
============================================================
Starting detection...

Running... Press Ctrl+C to stop
[DETECTION] Score: 0.847, Offset: 5.23 kHz, Voice: True
Buffers: 142, Events: 8
```

## Next Steps

1. **Start with simulation** to understand the system
2. **Fix libiio** for hardware use  
3. **Run integrated system** with full features
4. **Enable GPU** for better performance (optional)

## Getting Help

If you're stuck:

1. Run `python diagnose.py` for detailed diagnostics
2. Check the output of `./fix_libiio.sh`
3. Look at system logs: `dmesg | tail -50`
4. Verify Pluto+ firmware is up to date

The simulation mode (`alternative_detector.py --simulate`) should work **immediately** without any setup!
