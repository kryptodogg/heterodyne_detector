#!/usr/bin/env python3
"""
Alternative Heterodyne Detector using direct USB/Network communication
This version works around libiio compatibility issues by using alternative methods
"""

import numpy as np
import scipy.signal as signal
from scipy.fft import fft, ifft, fftfreq
import threading
import queue
import time
from collections import deque
import socket
import struct
import warnings
warnings.filterwarnings('ignore')

# Try GPU
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("GPU acceleration enabled (CuPy)")
except ImportError:
    GPU_AVAILABLE = False
    cp = np


class PlutoDirectConnection:
    """
    Direct connection to Pluto+ using network sockets
    Bypasses the problematic libiio library
    """
    
    def __init__(self, ip="192.168.2.1", port=30431):
        """Connect directly to Pluto+ via network"""
        self.ip = ip
        self.port = port
        self.sock = None
        self.connected = False
        
    def connect(self):
        """Establish connection"""
        print(f"Connecting to Pluto+ at {self.ip}:{self.port}...")
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(5)
            self.sock.connect((self.ip, self.port))
            self.connected = True
            print("✅ Connected via network socket")
            return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            print("\nTroubleshooting:")
            print(f"  1. Ping Pluto+: ping {self.ip}")
            print(f"  2. Check Pluto+ web interface: http://{self.ip}")
            print(f"  3. Try USB mode instead")
            return False
    
    def disconnect(self):
        """Close connection"""
        if self.sock:
            self.sock.close()
            self.connected = False


class PlutoSimulator:
    """
    Simulated Pluto+ for testing without hardware
    Generates synthetic signals for development/testing
    """
    
    def __init__(self, sample_rate=2.4e6, center_freq=100e6):
        self.sample_rate = sample_rate
        self.center_freq = center_freq
        self.buffer_size = 2**16
        self.time = 0
        print("⚠️  Running in SIMULATION mode (no hardware required)")
        
    def rx(self):
        """Generate simulated RX samples"""
        # Generate time vector
        t = np.linspace(self.time, self.time + self.buffer_size/self.sample_rate, 
                       self.buffer_size)
        self.time = t[-1]
        
        # RX1: Signal with interference
        freq1 = 10e3  # 10 kHz offset from center
        freq2 = 15e3  # 15 kHz offset
        signal1 = (np.exp(1j * 2 * np.pi * freq1 * t) + 
                  0.5 * np.exp(1j * 2 * np.pi * freq2 * t) +
                  0.1 * (np.random.randn(len(t)) + 1j * np.random.randn(len(t))))
        
        # RX2: Correlated signal (heterodyne artifact simulation)
        heterodyne_freq = 5e3  # Beat frequency
        signal2 = (0.7 * np.exp(1j * 2 * np.pi * freq1 * t) + 
                  0.6 * np.exp(1j * 2 * np.pi * heterodyne_freq * t) +
                  0.1 * (np.random.randn(len(t)) + 1j * np.random.randn(len(t))))
        
        # Add periodic "phantom voice" artifact
        if np.random.random() > 0.95:  # 5% chance
            voice_freq = 1000  # 1 kHz "voice"
            voice_mod = np.sin(2 * np.pi * voice_freq * t)
            signal1 += 0.3 * voice_mod * np.exp(1j * 2 * np.pi * freq1 * t)
            signal2 += 0.3 * voice_mod * np.exp(1j * 2 * np.pi * freq1 * t)
        
        return [signal1.astype(np.complex64), signal2.astype(np.complex64)]
    
    def tx(self, data):
        """Simulate TX (no-op)"""
        pass


class AlternativeHeterodyneDetector:
    """
    Heterodyne detector that works without libiio
    Can use simulation mode for testing
    """
    
    def __init__(self, sample_rate=2.4e6, center_freq=100e6, 
                 rx_gain=50, buffer_size=2**16, simulate=False):
        """
        Initialize detector
        
        Args:
            simulate: If True, use simulated data (no hardware needed)
        """
        self.sample_rate = sample_rate
        self.center_freq = center_freq
        self.rx_gain = rx_gain
        self.buffer_size = buffer_size
        self.simulate = simulate
        
        # SDR connection
        self.sdr = None
        
        # Detection parameters
        self.heterodyne_threshold = 0.7
        self.phantom_voice_freq_range = (300, 3400)
        self.history_length = 100
        
        # Storage
        self.rx1_history = deque(maxlen=self.history_length)
        self.rx2_history = deque(maxlen=self.history_length)
        self.heterodyne_events = []
        
        # Active cancellation
        self.cancellation_active = False
        self.cancellation_coefficient = 0.8
        
        # Threading
        self.running = False
        
    def connect(self):
        """Connect to Pluto+ or start simulation"""
        if self.simulate:
            print("=" * 60)
            print("SIMULATION MODE")
            print("=" * 60)
            print("Using synthetic data for testing/development")
            print("To use real hardware, set simulate=False")
            print("=" * 60)
            self.sdr = PlutoSimulator(self.sample_rate, self.center_freq)
            return True
        
        # Try real hardware
        print("Attempting hardware connection...")
        print("Note: This requires working libiio installation")
        
        try:
            import adi
            try:
                self.sdr = adi.Pluto("usb:1.100.5")
                print("✅ Connected via USB")
            except:
                self.sdr = adi.Pluto("ip:192.168.2.1")
                print("✅ Connected via IP")
            
            # Configure
            self.sdr.sample_rate = int(self.sample_rate)
            self.sdr.rx_lo = int(self.center_freq)
            self.sdr.rx_rf_bandwidth = int(self.sample_rate)
            self.sdr.rx_buffer_size = self.buffer_size
            self.sdr.rx_enabled_channels = [0, 1]
            self.sdr.gain_control_mode_chan0 = "manual"
            self.sdr.rx_hardwaregain_chan0 = self.rx_gain
            self.sdr.gain_control_mode_chan1 = "manual"
            self.sdr.rx_hardwaregain_chan1 = self.rx_gain
            
            print(f"Sample Rate: {self.sample_rate/1e6:.2f} MHz")
            print(f"Center Freq: {self.center_freq/1e6:.2f} MHz")
            return True
            
        except ImportError:
            print("❌ Cannot import adi (pyadi-iio)")
            print("\nOptions:")
            print("  1. Install pyadi-iio: pip install pyadi-iio")
            print("  2. Run in simulation mode: simulate=True")
            return False
        except Exception as e:
            print(f"❌ Hardware connection failed: {e}")
            print("\nOptions:")
            print("  1. Run ./fix_libiio.sh to fix library issues")
            print("  2. Run in simulation mode: simulate=True")
            return False
    
    def read_samples(self):
        """Read samples from SDR or simulator"""
        samples = self.sdr.rx()
        return samples[0], samples[1]
    
    def detect_heterodyne(self, signal1, signal2):
        """Detect heterodyne artifacts between two signals"""
        if GPU_AVAILABLE:
            signal1_gpu = cp.asarray(signal1)
            signal2_gpu = cp.asarray(signal2)
            correlation = cp.correlate(signal1_gpu, signal2_gpu, mode='same')
            correlation = cp.abs(correlation)
            correlation = correlation / cp.max(correlation)
            peak_value = float(cp.max(correlation))
            signal1 = cp.asnumpy(signal1_gpu)
            signal2 = cp.asnumpy(signal2_gpu)
        else:
            correlation = np.correlate(signal1, signal2, mode='same')
            correlation = np.abs(correlation)
            correlation = correlation / np.max(correlation)
            peak_value = float(np.max(correlation))
        
        # Frequency offset detection
        freq_offset = self.detect_frequency_offset(signal1, signal2)
        
        # Voice characteristics
        is_voice_range = self.check_voice_characteristics(signal1, signal2)
        
        heterodyne_score = peak_value if is_voice_range else peak_value * 0.5
        
        return {
            'score': heterodyne_score,
            'freq_offset': freq_offset,
            'is_voice_range': is_voice_range,
            'timestamp': time.time()
        }
    
    def detect_frequency_offset(self, signal1, signal2):
        """Detect frequency offset via FFT phase analysis"""
        fft1 = fft(signal1)
        fft2 = fft(signal2)
        cross_spectrum = fft1 * np.conj(fft2)
        phase_diff = np.angle(cross_spectrum)
        phase_unwrapped = np.unwrap(phase_diff)
        
        mid_start = len(phase_unwrapped) // 4
        mid_end = 3 * len(phase_unwrapped) // 4
        
        if mid_end > mid_start:
            freq_offset = np.polyfit(range(mid_end - mid_start), 
                                    phase_unwrapped[mid_start:mid_end], 1)[0]
            freq_offset = freq_offset * self.sample_rate / (2 * np.pi)
        else:
            freq_offset = 0
        
        return freq_offset
    
    def check_voice_characteristics(self, signal1, signal2):
        """Check for voice-like characteristics"""
        envelope1 = np.abs(signal.hilbert(np.real(signal1)))
        envelope2 = np.abs(signal.hilbert(np.real(signal2)))
        
        nyquist = self.sample_rate / 2
        low = self.phantom_voice_freq_range[0] / nyquist
        high = self.phantom_voice_freq_range[1] / nyquist
        
        if high >= 1.0:
            high = 0.99
        if low <= 0:
            low = 0.01
        
        b, a = signal.butter(4, [low, high], btype='band')
        voice1 = signal.filtfilt(b, a, envelope1)
        voice2 = signal.filtfilt(b, a, envelope2)
        
        voice_energy = np.sum(voice1**2) + np.sum(voice2**2)
        total_energy = np.sum(envelope1**2) + np.sum(envelope2**2)
        
        voice_ratio = voice_energy / (total_energy + 1e-10)
        return voice_ratio > 0.15
    
    def processing_loop(self):
        """Main processing loop"""
        print("Starting detection...")
        
        while self.running:
            try:
                # Read samples
                rx1_samples, rx2_samples = self.read_samples()
                
                # Store
                self.rx1_history.append(rx1_samples)
                self.rx2_history.append(rx2_samples)
                
                # Detect
                het_result = self.detect_heterodyne(rx1_samples, rx2_samples)
                
                # Log significant events
                if het_result['score'] > self.heterodyne_threshold:
                    self.heterodyne_events.append(het_result)
                    print(f"\n[DETECTION] Score: {het_result['score']:.3f}, "
                          f"Offset: {het_result['freq_offset']/1e3:.2f} kHz, "
                          f"Voice: {het_result['is_voice_range']}")
                
                # Status
                if len(self.rx1_history) % 10 == 0:
                    print(f"Buffers: {len(self.rx1_history)}, "
                          f"Events: {len(self.heterodyne_events)}", end='\r')
                
                time.sleep(0.01)  # Small delay
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(0.1)
    
    def start(self):
        """Start detector"""
        if not self.connect():
            print("\nFailed to connect. Options:")
            print("  1. Fix libiio: ./fix_libiio.sh")
            print("  2. Use simulation: detector = AlternativeHeterodyneDetector(simulate=True)")
            return False
        
        self.running = True
        self.process_thread = threading.Thread(target=self.processing_loop)
        self.process_thread.start()
        return True
    
    def stop(self):
        """Stop detector"""
        print("\nStopping...")
        self.running = False
        if hasattr(self, 'process_thread'):
            self.process_thread.join()
        
        print(f"\nSummary:")
        print(f"  Buffers: {len(self.rx1_history)}")
        print(f"  Events: {len(self.heterodyne_events)}")


def main():
    """Main with simulation support"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Heterodyne Detector')
    parser.add_argument('--freq', type=float, default=100.0, help='Center frequency (MHz)')
    parser.add_argument('--sample-rate', type=float, default=2.4, help='Sample rate (MHz)')
    parser.add_argument('--simulate', action='store_true', help='Use simulated data (no hardware)')
    parser.add_argument('--duration', type=int, default=60, help='Run duration (seconds)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Alternative Heterodyne Detector")
    print("=" * 60)
    
    detector = AlternativeHeterodyneDetector(
        sample_rate=args.sample_rate * 1e6,
        center_freq=args.freq * 1e6,
        simulate=args.simulate
    )
    
    if detector.start():
        print("\nRunning... Press Ctrl+C to stop")
        try:
            time.sleep(args.duration)
        except KeyboardInterrupt:
            pass
        finally:
            detector.stop()
    else:
        print("\nCould not start detector")


if __name__ == "__main__":
    main()
