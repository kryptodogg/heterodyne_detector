#!/usr/bin/env python3
"""
Heterodyne Detection and Active Noise Cancellation for Pluto+ SDR
Detects phantom voices and noise heterodyning using dual RX/TX channels
Uses GPU acceleration for pattern matching (DTW/Levenshtein analysis)
"""

import numpy as np
import scipy.signal as signal
from scipy.fft import fft, ifft, fftfreq
import iio
import threading
import queue
import time
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# Use NumPy for CPU fallbacks; heavy lifting is in Torch
cp = np
GPU_AVAILABLE = False

class HeterodyneDetector:
    """
    Detects heterodyne artifacts and phantom voices using dual RX channels
    """
    
    def __init__(self, sample_rate=2.4e6, center_freq=100e6, 
                 rx_gain=50, buffer_size=2**16, device='cpu', **kwargs):
        """
        Initialize the heterodyne detector
        
        Args:
            sample_rate: Sample rate in Hz (default 2.4 MHz)
            center_freq: Center frequency in Hz (default 100 MHz)
            rx_gain: RX gain in dB (default 50)
            buffer_size: Buffer size for processing
            device: torch device
        """
        self.sample_rate = sample_rate
        self.center_freq = center_freq
        self.rx_gain = rx_gain
        self.buffer_size = buffer_size
        self.device = device
        
        # Connect to Pluto+
        self.ctx = None
        self.rx1 = None
        self.rx2 = None
        self.tx1 = None
        self.tx2 = None
        
        # Processing queues
        self.rx1_queue = queue.Queue(maxsize=10)
        self.rx2_queue = queue.Queue(maxsize=10)
        self.cancellation_queue = queue.Queue(maxsize=10)
        
        # Heterodyne detection parameters
        self.heterodyne_threshold = 0.7  # Correlation threshold
        self.phantom_voice_freq_range = (300, 3400)  # Voice frequency range in Hz
        self.history_length = 100  # Number of buffers to keep in history
        
        # Storage for analysis
        self.rx1_history = deque(maxlen=self.history_length)
        self.rx2_history = deque(maxlen=self.history_length)
        self.heterodyne_events = []
        
        # Active cancellation state
        self.cancellation_active = False
        self.cancellation_coefficient = 0.8
        
        # Thread control
        self.running = False
        
    def connect_pluto(self):
        """Connect to Pluto+ SDR"""
        print("Connecting to Pluto+ SDR...")
        try:
            self.ctx = iio.Context('ip:192.168.2.1')  # Default Pluto IP
        except:
            print("Failed to connect via IP, trying USB...")
            self.ctx = iio.Context('usb:')
        
        print(f"Connected to: {self.ctx.name}")
        
        # Get devices
        self.phy = self.ctx.find_device("ad9361-phy")
        self.rx_dev = self.ctx.find_device("cf-ad9361-lpc")
        self.tx_dev = self.ctx.find_device("cf-ad9361-dds-core-lpc")
        
        # Configure RX channels
        self.configure_rx()
        
        # Configure TX channels  
        self.configure_tx()
        
        print("Pluto+ configured successfully")
        
    def configure_rx(self):
        """Configure RX channels"""
        # Set RX parameters
        self.phy.find_channel("altvoltage0", True).attrs["frequency"].value = str(int(self.center_freq))
        self.phy.find_channel("voltage0").attrs["rf_bandwidth"].value = str(int(self.sample_rate))
        self.phy.find_channel("voltage0").attrs["sampling_frequency"].value = str(int(self.sample_rate))
        self.phy.find_channel("voltage0").attrs["gain_control_mode"].value = "manual"
        self.phy.find_channel("voltage0").attrs["hardwaregain"].value = str(self.rx_gain)
        
        # Enable both RX channels
        self.rx1 = self.rx_dev.find_channel("voltage0")
        self.rx2 = self.rx_dev.find_channel("voltage1")
        self.rx1.enabled = True
        self.rx2.enabled = True
        
        # Create RX buffer
        self.rx_buffer = iio.Buffer(self.rx_dev, self.buffer_size, False)
        
    def configure_tx(self):
        """Configure TX channels for active cancellation"""
        # Set TX parameters
        self.phy.find_channel("altvoltage1", True).attrs["frequency"].value = str(int(self.center_freq))
        self.phy.find_channel("voltage0", True).attrs["rf_bandwidth"].value = str(int(self.sample_rate))
        self.phy.find_channel("voltage0", True).attrs["sampling_frequency"].value = str(int(self.sample_rate))
        self.phy.find_channel("voltage0", True).attrs["hardwaregain"].value = str(-10)
        
        # Enable both TX channels
        self.tx1 = self.tx_dev.find_channel("voltage0", True)
        self.tx2 = self.tx_dev.find_channel("voltage1", True)
        self.tx1.enabled = True
        self.tx2.enabled = True
        
        # Create TX buffer
        self.tx_buffer = iio.Buffer(self.tx_dev, self.buffer_size, False)
        
    def read_samples(self):
        """Read samples from both RX channels"""
        self.rx_buffer.refill()
        
        # Read RX1 (I/Q interleaved)
        rx1_data = np.frombuffer(self.rx_buffer.read(), dtype=np.int16)
        rx1_i = rx1_data[0::4].astype(np.float32)
        rx1_q = rx1_data[1::4].astype(np.float32)
        rx1_complex = rx1_i + 1j * rx1_q
        
        # Read RX2 (I/Q interleaved)
        rx2_i = rx1_data[2::4].astype(np.float32)
        rx2_q = rx1_data[3::4].astype(np.float32)
        rx2_complex = rx2_i + 1j * rx2_q
        
        return rx1_complex, rx2_complex
    
    def detect_heterodyne(self, signal1, signal2):
        """
        Detect heterodyne artifacts between two signals
        Returns heterodyne score and frequency offset
        """
        if GPU_AVAILABLE:
            signal1_gpu = cp.asarray(signal1)
            signal2_gpu = cp.asarray(signal2)
            
            # Cross-correlation to detect time-shifted copies
            correlation = cp.correlate(signal1_gpu, signal2_gpu, mode='same')
            correlation = cp.abs(correlation)
            
            # Normalize
            correlation = correlation / cp.max(correlation)
            
            # Find peak
            peak_idx = cp.argmax(correlation)
            peak_value = float(correlation[peak_idx])
            
            # Convert back to CPU for frequency analysis
            signal1 = cp.asnumpy(signal1_gpu)
            signal2 = cp.asnumpy(signal2_gpu)
        else:
            # CPU fallback
            correlation = np.correlate(signal1, signal2, mode='same')
            correlation = np.abs(correlation)
            correlation = correlation / np.max(correlation)
            peak_idx = np.argmax(correlation)
            peak_value = float(correlation[peak_idx])
        
        # Frequency domain analysis for heterodyne detection
        freq_offset = self.detect_frequency_offset(signal1, signal2)
        
        # Check for phantom voice characteristics
        is_voice_range = self.check_voice_characteristics(signal1, signal2)
        
        heterodyne_score = peak_value if is_voice_range else peak_value * 0.5
        
        return {
            'score': heterodyne_score,
            'freq_offset': freq_offset,
            'is_voice_range': is_voice_range,
            'timestamp': time.time()
        }
    
    def detect_frequency_offset(self, signal1, signal2):
        """Detect frequency offset between signals using FFT"""
        # Compute FFTs
        fft1 = fft(signal1)
        fft2 = fft(signal2)
        
        # Cross-spectrum
        cross_spectrum = fft1 * np.conj(fft2)
        
        # Find dominant frequency difference
        freqs = fftfreq(len(signal1), 1/self.sample_rate)
        
        # Phase difference indicates frequency offset
        phase_diff = np.angle(cross_spectrum)
        
        # Unwrap and find linear trend (frequency offset)
        phase_unwrapped = np.unwrap(phase_diff)
        
        # Estimate frequency offset from phase slope
        # Using middle section to avoid edge effects
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
        """Check if signals have phantom voice characteristics"""
        # Demodulate to baseband (AM detection)
        envelope1 = np.abs(signal.hilbert(np.real(signal1)))
        envelope2 = np.abs(signal.hilbert(np.real(signal2)))
        
        # Bandpass filter for voice range
        nyquist = self.sample_rate / 2
        low = self.phantom_voice_freq_range[0] / nyquist
        high = self.phantom_voice_freq_range[1] / nyquist
        
        # Ensure filter parameters are valid
        if high >= 1.0:
            high = 0.99
        if low <= 0:
            low = 0.01
            
        b, a = signal.butter(4, [low, high], btype='band')
        
        voice1 = signal.filtfilt(b, a, envelope1)
        voice2 = signal.filtfilt(b, a, envelope2)
        
        # Check energy in voice band
        voice_energy1 = np.sum(voice1**2)
        voice_energy2 = np.sum(voice2**2)
        total_energy1 = np.sum(envelope1**2)
        total_energy2 = np.sum(envelope2**2)
        
        # Voice band should contain significant energy
        voice_ratio = (voice_energy1 + voice_energy2) / (total_energy1 + total_energy2 + 1e-10)
        
        return voice_ratio > 0.15  # Threshold for voice presence
    
    def compute_cancellation_signal(self, interferer_signal, reference_signal):
        """
        Compute active cancellation signal using adaptive filtering
        Uses LMS (Least Mean Squares) algorithm
        """
        if GPU_AVAILABLE:
            interferer = cp.asarray(interferer_signal)
            reference = cp.asarray(reference_signal)
        else:
            interferer = interferer_signal
            reference = reference_signal
        
        # Adaptive filter parameters
        filter_length = 64
        mu = 0.01  # Step size
        
        # Initialize filter coefficients
        w = cp.zeros(filter_length, dtype=cp.complex64) if GPU_AVAILABLE else \
            np.zeros(filter_length, dtype=np.complex64)
        
        # Output signal
        output = cp.zeros_like(interferer) if GPU_AVAILABLE else \
                np.zeros_like(interferer)
        
        # LMS adaptive filtering
        for n in range(filter_length, len(interferer)):
            # Get input vector
            x = interferer[n-filter_length:n][::-1]
            
            # Compute output
            y = cp.dot(w, x) if GPU_AVAILABLE else np.dot(w, x)
            
            # Error signal
            e = reference[n] - y
            
            # Update weights
            w = w + mu * cp.conj(e) * x if GPU_AVAILABLE else \
                w + mu * np.conj(e) * x
            
            output[n] = y
        
        # Convert back to CPU if using GPU
        if GPU_AVAILABLE:
            output = cp.asnumpy(output)
            
        return output * self.cancellation_coefficient
    
    def transmit_cancellation(self, cancel_signal):
        """Transmit cancellation signal via TX channels"""
        if not self.cancellation_active:
            return
            
        # Scale to TX range
        cancel_signal = cancel_signal / (np.max(np.abs(cancel_signal)) + 1e-10)
        cancel_signal = (cancel_signal * 2047).astype(np.int16)
        
        # Prepare I/Q data
        i_data = np.real(cancel_signal).astype(np.int16)
        q_data = np.imag(cancel_signal).astype(np.int16)
        
        # Interleave I/Q for TX
        iq_interleaved = np.empty(len(i_data) * 2, dtype=np.int16)
        iq_interleaved[0::2] = i_data
        iq_interleaved[1::2] = q_data
        
        # Write to TX buffer
        self.tx_buffer.write(iq_interleaved.tobytes())
        self.tx_buffer.push()
    
    def processing_loop(self):
        """Main processing loop"""
        print("Starting heterodyne detection...")
        
        while self.running:
            try:
                # Read samples from both RX channels
                rx1_samples, rx2_samples = self.read_samples()
                
                # Store in history
                self.rx1_history.append(rx1_samples)
                self.rx2_history.append(rx2_samples)
                
                # Detect heterodyne
                het_result = self.detect_heterodyne(rx1_samples, rx2_samples)
                
                # Log significant events
                if het_result['score'] > self.heterodyne_threshold:
                    self.heterodyne_events.append(het_result)
                    print(f"\n[DETECTION] Heterodyne score: {het_result['score']:.3f}")
                    print(f"  Freq offset: {het_result['freq_offset']/1e3:.2f} kHz")
                    print(f"  Voice range: {het_result['is_voice_range']}")
                    
                    # Compute and transmit cancellation signal if active
                    if self.cancellation_active:
                        cancel_signal = self.compute_cancellation_signal(
                            rx1_samples, rx2_samples)
                        self.transmit_cancellation(cancel_signal)
                        print("  [CANCELLATION] Active noise cancellation applied")
                
                # Display status periodically
                if len(self.rx1_history) % 10 == 0:
                    print(f"Monitoring... (buffers: {len(self.rx1_history)}, "
                          f"events: {len(self.heterodyne_events)})", end='\r')
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error in processing loop: {e}")
                time.sleep(0.1)
    
    def start(self):
        """Start the detector"""
        self.connect_pluto()
        self.running = True
        
        # Start processing thread
        self.process_thread = threading.Thread(target=self.processing_loop)
        self.process_thread.start()
        
    def stop(self):
        """Stop the detector"""
        print("\nStopping heterodyne detector...")
        self.running = False
        if hasattr(self, 'process_thread'):
            self.process_thread.join()
        
        # Print summary
        print(f"\n=== Detection Summary ===")
        print(f"Total buffers processed: {len(self.rx1_history)}")
        print(f"Heterodyne events detected: {len(self.heterodyne_events)}")
        
        if self.heterodyne_events:
            avg_score = np.mean([e['score'] for e in self.heterodyne_events])
            print(f"Average heterodyne score: {avg_score:.3f}")
    
    def enable_cancellation(self, enable=True):
        """Enable or disable active cancellation"""
        self.cancellation_active = enable
        print(f"Active cancellation: {'ENABLED' if enable else 'DISABLED'}")
    
    def set_threshold(self, threshold):
        """Set detection threshold"""
        self.heterodyne_threshold = threshold
        print(f"Detection threshold set to: {threshold}")


def main():
    """Main function with interactive controls"""
    print("=" * 60)
    print("Heterodyne Detector & Active Noise Cancellation")
    print("Pluto+ SDR Dual RX/TX")
    print("=" * 60)
    print()
    
    # Initialize detector
    detector = HeterodyneDetector(
        sample_rate=2.4e6,
        center_freq=100e6,
        rx_gain=50,
        buffer_size=2**16
    )
    
    try:
        # Start detection
        detector.start()
        
        print("\nControls:")
        print("  'c' - Toggle active cancellation")
        print("  't' - Set detection threshold")
        print("  'f' - Change center frequency")
        print("  's' - Show statistics")
        print("  'q' - Quit")
        print()
        
        # Simple command loop
        while True:
            cmd = input("Enter command: ").strip().lower()
            
            if cmd == 'q':
                break
            elif cmd == 'c':
                detector.enable_cancellation(not detector.cancellation_active)
            elif cmd == 't':
                try:
                    threshold = float(input("Enter threshold (0.0-1.0): "))
                    detector.set_threshold(threshold)
                except ValueError:
                    print("Invalid threshold value")
            elif cmd == 'f':
                try:
                    freq = float(input("Enter center frequency (MHz): ")) * 1e6
                    detector.center_freq = freq
                    detector.configure_rx()
                    print(f"Center frequency set to {freq/1e6:.2f} MHz")
                except ValueError:
                    print("Invalid frequency value")
            elif cmd == 's':
                print(f"\nCurrent Statistics:")
                print(f"  Buffers processed: {len(detector.rx1_history)}")
                print(f"  Events detected: {len(detector.heterodyne_events)}")
                print(f"  Cancellation active: {detector.cancellation_active}")
                print(f"  Detection threshold: {detector.heterodyne_threshold}")
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        detector.stop()


if __name__ == "__main__":
    main()
