#!/bin/bash
# Fix script for libiio version mismatch
# This script rebuilds libiio from source to match the Python bindings

set -e

echo "============================================================"
echo "libiio Fix Script for Pluto+ SDR"
echo "============================================================"
echo ""
echo "This script will:"
echo "  1. Build and install libiio from source"
echo "  2. Install compatible Python bindings"
echo "  3. Test the installation"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
fi

echo ""
echo "Step 1: Installing build dependencies..."
sudo apt update
sudo apt install -y build-essential cmake git libxml2-dev bison flex \
    libcdk5-dev libusb-1.0-0-dev libserialport-dev libavahi-client-dev \
    libaio-dev python3-dev python3-setuptools

echo ""
echo "Step 2: Removing old libiio installations..."
# Remove old Python bindings
pip uninstall -y pylibiio iio 2>/dev/null || true

# Remove old system library (backup first)
if [ -f /usr/local/lib/libiio.so ]; then
    sudo mv /usr/local/lib/libiio.so /usr/local/lib/libiio.so.backup 2>/dev/null || true
fi

echo ""
echo "Step 3: Building libiio from source..."
cd /tmp
rm -rf libiio
git clone https://github.com/analogdevicesinc/libiio.git
cd libiio

# Checkout a stable version
git checkout v0.25

mkdir -p build
cd build

cmake .. \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DPYTHON_BINDINGS=ON \
    -DWITH_USB_BACKEND=ON \
    -DWITH_SERIAL_BACKEND=ON \
    -DWITH_NETWORK_BACKEND=ON

make -j$(nproc)

echo ""
echo "Step 4: Installing libiio..."
sudo make install
sudo ldconfig

echo ""
echo "Step 5: Setting up udev rules for Pluto+..."
sudo bash -c 'cat > /etc/udev/rules.d/53-adi-plutosdr-usb.rules << EOF
# PlutoSDR
SUBSYSTEM=="usb", ATTRS{idVendor}=="0456", ATTRS{idProduct}=="b673", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="usb", ATTRS{idVendor}=="0456", ATTRS{idProduct}=="b674", MODE="0666", GROUP="plugdev"
# M2k
SUBSYSTEM=="usb", ATTRS{idVendor}=="0456", ATTRS{idProduct}=="b671", MODE="0666", GROUP="plugdev"
EOF'

sudo udevadm control --reload-rules
sudo udevadm trigger

# Add user to plugdev group
sudo usermod -a -G plugdev $USER

echo ""
echo "Step 6: Installing Python bindings..."
cd /tmp/libiio/bindings/python
python3 setup.py install

echo ""
echo "Step 7: Installing pyadi-iio..."
pip install pyadi-iio

echo ""
echo "Step 8: Testing installation..."
echo ""

# Test libiio
if command -v iio_info &> /dev/null; then
    echo "✅ iio_info command available"
    echo "Testing USB detection..."
    iio_info -u || echo "⚠️  No devices found (this is OK if Pluto+ not connected)"
else
    echo "⚠️  iio_info not in PATH, adding /usr/local/bin"
    export PATH="/usr/local/bin:$PATH"
fi

# Test Python import
python3 << 'PYTHON_TEST'
import sys
print("\nTesting Python imports...")

try:
    import iio
    print("✅ iio module imports successfully")
    
    # Test the problematic function
    try:
        import ctypes
        lib = iio._lib
        count = lib.iio_get_backends_count
        print("✅ iio_get_backends_count is available")
    except AttributeError as e:
        print(f"❌ Still missing iio_get_backends_count: {e}")
        sys.exit(1)
        
except ImportError as e:
    print(f"❌ Cannot import iio: {e}")
    sys.exit(1)

try:
    import adi
    print("✅ adi (pyadi-iio) imports successfully")
except ImportError as e:
    print(f"❌ Cannot import adi: {e}")
    print("Try: pip install pyadi-iio")
    sys.exit(1)

print("\n✅ All Python bindings working!")
PYTHON_TEST

echo ""
echo "============================================================"
echo "Installation Complete!"
echo "============================================================"
echo ""
echo "⚠️  IMPORTANT: You may need to:"
echo "  1. Log out and back in (for group membership)"
echo "  2. Reconnect your Pluto+ device"
echo "  3. Run: source ~/.bashrc"
echo ""
echo "Test your installation with:"
echo "  python diagnose.py"
echo ""
echo "Or test Pluto+ directly:"
echo "  iio_info -u"
echo "  python -c 'import adi; sdr = adi.Pluto(); print(\"Connected!\")'"
echo ""
