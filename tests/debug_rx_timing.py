import time
import adi
import numpy as np

def test_rx_timing():
    uri = "ip:192.168.2.1"
    print(f"Connecting to {uri}...")
    sdr = adi.ad9361(uri=uri)
    sdr.sample_rate = int(10e6)
    sdr.rx_lo = int(2.4e9)
    sdr.rx_enabled_channels = [0, 1]
    sdr.rx_buffer_size = 2**16
    
    print("Capturing 10 buffers...")
    times = []
    for i in range(10):
        start = time.time()
        samples = sdr.rx()
        end = time.time()
        times.append(end - start)
        print(f"  Buffer {i}: {(end-start)*1000:.2f} ms")
    
    print(f"\nAverage RX time: {np.mean(times)*1000:.2f} ms")
    sdr.close()

if __name__ == "__main__":
    test_rx_timing()

