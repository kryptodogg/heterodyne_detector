import adi
import time
import numpy as np

print("Testing Pluto+ Connectivity...")
try:
    sdr = adi.Pluto("ip:192.168.2.1")
    print("✅ Context created")
    
    sdr.sample_rate = int(2.4e6)
    sdr.rx_lo = int(100e6)
    sdr.rx_enabled_channels = [0]
    sdr.rx_buffer_size = 1024*16
    
    print("Attempting to receive samples...")
    start = time.time()
    samples = sdr.rx()
    end = time.time()
    
    if samples is not None and len(samples) > 0:
        print(f"✅ Received {len(samples)} samples in {end-start:.4f}s")
        print(f"   First 5 samples: {samples[:5]}")
    else:
        print("❌ Received empty samples")
        
except Exception as e:
    print(f"❌ Error: {e}")
