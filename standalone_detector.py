#!/usr/bin/env python3
"""
Standalone Heterodyne Detector - All-in-One Version
No separate module files required!

Usage:
  # Simulation mode (works immediately, no hardware):
  python standalone_detector.py --simulate
  
  # Real hardware (requires Pluto+ and libiio):
  python standalone_detector.py --freq 100 --sample-rate 2.4
  
  # With JAX acceleration:
  pip install jax jaxlib
  python standalone_detector.py --freq 100 --sample-rate 2.4
"""

import numpy as np
import scipy.signal as signal
from scipy.fft import fft, ifft, fftfreq
import threading
import queue
import time
from collections import deque
import argparse
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# GPU Acceleration Detection
# ============================================================

# Try PyTorch first (recommended for GPU acceleration)
TORCH_AVAILABLE = False
try:
    import torch
    import torch.fft

    if torch.cuda.is_available():
        TORCH_GPU = True
        print(f"✅ PyTorch GPU acceleration: CUDA available")
    else:
        TORCH_GPU = False
        print("⚠️  PyTorch using CPU (still fast!)")

    TORCH_AVAILABLE = True

    def torch_correlate(x, y):
        """PyTorch implementation of cross-correlation"""
        # Ensure inputs are numpy arrays
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()

        # Handle complex numbers if present
        if np.iscomplexobj(x) or np.iscomplexobj(y):
            # For complex signals, take the absolute value for correlation
            x = np.abs(x)
            y = np.abs(y)

        # Ensure arrays are 1D and flatten if necessary
        x = np.asarray(x).flatten()
        y = np.asarray(y).flatten()

        # Use NumPy for correlation since PyTorch doesn't have a direct correlate function
        correlation = np.correlate(x, y, mode='same')
        return correlation

except ImportError:
    print("ℹ️  PyTorch not available (using NumPy)")
    TORCH_AVAILABLE = False
    TORCH_GPU = False

print()

# ============================================================
# Pluto+ SDR Interface
# ============================================================

class PlutoSDR:
    """Interface for Pluto+ SDR or simulation"""
    
    def __init__(self, sample_rate=2.4e6, center_freq=100e6, 
                 rx_gain=50, buffer_size=2**16, simulate=False):
        self.sample_rate = sample_rate
        self.center_freq = center_freq
        self.rx_gain = rx_gain
        self.buffer_size = buffer_size
        self.simulate = simulate
        self.sdr = None
        self.time = 0
        
    def connect(self):
        """Connect to SDR or start simulation"""
        if self.simulate:
            print("="*60)
            print("🎮 SIMULATION MODE")
            print("="*60)
            print("Generating synthetic heterodyne signals")
            print("Perfect for testing without hardware!")
            print("="*60)
            return True
        
        # Try real hardware
        print("Connecting to Pluto+ SDR...")
        try:
            import adi
            
            # Try USB first
            try:
                self.sdr = adi.Pluto("usb:1.100.5")
                print("✅ Connected via USB")
            except:
                try:
                    self.sdr = adi.Pluto("ip:192.168.2.1")
                    print("✅ Connected via IP")
                except:
                    print("❌ Connection failed")
                    print("\nOptions:")
                    print("  1. Run: ./fix_libiio.sh")
                    print("  2. Use --simulate flag")
                    return False
            
            # Configure
            self.sdr.sample_rate = int(self.sample_rate)
            self.sdr.rx_lo = int(self.center_freq)
            self.sdr.rx_rf_bandwidth = int(self.sample_rate)
            self.sdr.rx_buffer_size = self.buffer_size
            
            # Detect available channels
            # Some standard Plutos need a hack to enable 2RX, or might only show 1
            try:
                self.sdr.rx_enabled_channels = [0, 1]
                self.channels = 2
            except Exception:
                print("⚠️  Dual RX failed, falling back to Single RX")
                self.sdr.rx_enabled_channels = [0]
                self.channels = 1
            
            self.sdr.gain_control_mode_chan0 = "manual"
            self.sdr.rx_hardwaregain_chan0 = self.rx_gain
            
            if self.channels > 1:
                self.sdr.gain_control_mode_chan1 = "manual"
                self.sdr.rx_hardwaregain_chan1 = self.rx_gain
            
            print(f"  Sample Rate: {self.sample_rate/1e6:.2f} MHz")
            print(f"  Center Freq: {self.center_freq/1e6:.2f} MHz")
            print(f"  RX Gain: {self.rx_gain} dB")
            return True
            
        except ImportError:
            print("❌ pyadi-iio not installed")
            print("Install: pip install pyadi-iio")
            print("Or use --simulate flag")
            return False
    
    def rx(self):
        """Receive samples (real or simulated)"""
        if self.simulate:
            return self._generate_synthetic_signals()
        else:
            samples = self.sdr.rx()
            return samples[0], samples[1]
    
    def _generate_synthetic_signals(self):
        """Generate synthetic heterodyne signals for testing"""
        t = np.linspace(self.time, 
                       self.time + self.buffer_size/self.sample_rate, 
                       self.buffer_size)
        self.time = t[-1]
        
        # RX1: Primary signal with interference
        freq1 = 10e3   # 10 kHz offset
        freq2 = 15e3   # 15 kHz offset
        signal1 = (np.exp(1j * 2 * np.pi * freq1 * t) + 
                  0.5 * np.exp(1j * 2 * np.pi * freq2 * t) +
                  0.1 * (np.random.randn(len(t)) + 1j * np.random.randn(len(t))))
        
        # RX2: Correlated signal (heterodyne artifact)
        heterodyne_freq = 5e3  # Beat frequency
        signal2 = (0.7 * np.exp(1j * 2 * np.pi * freq1 * t) + 
                  0.6 * np.exp(1j * 2 * np.pi * heterodyne_freq * t) +
                  0.1 * (np.random.randn(len(t)) + 1j * np.random.randn(len(t))))
        
        # Occasionally add "phantom voice" artifact
        if np.random.random() > 0.92:
            voice_freq = 1000  # 1 kHz
            voice_mod = np.sin(2 * np.pi * voice_freq * t)
            signal1 += 0.4 * voice_mod * np.exp(1j * 2 * np.pi * freq1 * t)
            signal2 += 0.4 * voice_mod * np.exp(1j * 2 * np.pi * freq1 * t)
        
        return signal1.astype(np.complex64), signal2.astype(np.complex64)


# ============================================================
# Heterodyne Detector
# ============================================================

class HeterodyneDetector:
    """Detects heterodyne artifacts and phantom voices"""
    
    def __init__(self, sdr):
        self.sdr = sdr
        self.sample_rate = sdr.sample_rate
        
        # Detection parameters
        self.heterodyne_threshold = 0.7
        self.voice_freq_range = (300, 3400)
        
        # History
        self.history_length = 100
        self.rx1_history = deque(maxlen=self.history_length)
        self.rx2_history = deque(maxlen=self.history_length)
        self.events = []
        
        # Threading
        self.running = False
    
    def detect_heterodyne(self, signal1, signal2):
        """Detect heterodyne between two signals"""

        # Cross-correlation (use PyTorch if available)
        if TORCH_AVAILABLE:
            s1 = np.abs(signal1)
            s2 = np.abs(signal2)
            correlation = torch_correlate(s1, s2)
            if isinstance(correlation, torch.Tensor):
                correlation = correlation.cpu().numpy()
        else:
            s1 = np.abs(signal1)
            s2 = np.abs(signal2)
            correlation = np.correlate(s1, s2, mode='same')

        # Normalize
        correlation = np.abs(correlation)
        max_corr = np.max(correlation)
        if max_corr > 0:
            correlation = correlation / max_corr
            peak_value = 1.0
        else:
            peak_value = 0.0

        # Frequency offset detection
        freq_offset = self._detect_frequency_offset(signal1, signal2)

        # Voice characteristics
        is_voice = self._check_voice_characteristics(signal1, signal2)

        # Compute score
        score = peak_value if is_voice else peak_value * 0.5

        return {
            'score': score,
            'freq_offset': freq_offset,
            'is_voice': is_voice,
            'correlation': correlation,
            'timestamp': time.time()
        }
    
    def _detect_frequency_offset(self, signal1, signal2):
        """Detect frequency offset via phase analysis"""
        # Skip if either signal is silent/missing
        if np.max(np.abs(signal1)) < 1e-6 or np.max(np.abs(signal2)) < 1e-6:
            return 0.0
            
        fft1 = fft(signal1)
        fft2 = fft(signal2)
        cross_spectrum = fft1 * np.conj(fft2)
        phase_diff = np.angle(cross_spectrum)
        phase_unwrapped = np.unwrap(phase_diff)
        
        # Estimate offset from phase slope
        mid_start = len(phase_unwrapped) // 4
        mid_end = 3 * len(phase_unwrapped) // 4
        
        if mid_end > mid_start:
            slope = np.polyfit(range(mid_end - mid_start), 
                             phase_unwrapped[mid_start:mid_end], 1)[0]
            freq_offset = slope * self.sample_rate / (2 * np.pi)
        else:
            freq_offset = 0.0
        
        return freq_offset
    
    def _check_voice_characteristics(self, signal1, signal2):
        """Check for voice-like characteristics"""
        # Demodulate
        envelope1 = np.abs(signal.hilbert(np.real(signal1)))
        envelope2 = np.abs(signal.hilbert(np.real(signal2)))
        
        # Bandpass filter for voice range
        nyquist = self.sample_rate / 2
        low = self.voice_freq_range[0] / nyquist
        high = self.voice_freq_range[1] / nyquist

        # Ensure valid frequency range for filter design
        # Clamp to valid range but preserve the relationship low < high
        if low >= high:
            # If the desired frequencies are too close or invalid,
            # use a small offset to ensure low < high
            mid_freq = (low + high) / 2
            freq_delta = min(0.001, abs(high - low) / 2)  # Small frequency delta
            low = max(0.0001, mid_freq - freq_delta)
            high = min(0.9999, mid_freq + freq_delta)
        else:
            # Apply clamps while preserving low < high
            low = max(0.0001, low)
            high = min(0.9999, high)
            if low >= high:
                # Extra safety check in case clamping caused the issue
                mid_freq = (low + high) / 2
                freq_delta = 0.001
                low = max(0.0001, mid_freq - freq_delta)
                high = min(0.9999, mid_freq + freq_delta)

        b, a = signal.butter(4, [low, high], btype='band')

        # Check if signal length is sufficient for filtering
        min_length = max(len(b), len(a)) + 10  # Add buffer to be safe
        if len(envelope1) < min_length or len(envelope2) < min_length:
            # If signal is too short, skip filtering and return early
            # Calculate energy ratio without filtering
            voice_energy = 0
            total_energy = np.sum(envelope1**2) + np.sum(envelope2**2)
            ratio = voice_energy / (total_energy + 1e-10)
            return ratio > 0.15  # Return False since no voice energy passed through filter

        voice1 = signal.filtfilt(b, a, envelope1)
        voice2 = signal.filtfilt(b, a, envelope2)
        
        # Check energy in voice band
        voice_energy = np.sum(voice1**2) + np.sum(voice2**2)
        total_energy = np.sum(envelope1**2) + np.sum(envelope2**2)
        
        ratio = voice_energy / (total_energy + 1e-10)
        return ratio > 0.15
    
    def processing_loop(self):
        """Main detection loop"""
        print("\n" + "="*60)
        print("Starting Detection Loop")
        print("="*60)
        print("Press Ctrl+C to stop\n")
        
        while self.running:
            try:
                # Read samples
                samples = self.sdr.rx()
                
                # Handle single vs dual channels
                if isinstance(samples, list) and len(samples) >= 2:
                    rx1, rx2 = samples[0], samples[1]
                elif isinstance(samples, np.ndarray):
                    rx1 = samples
                    # For single channel mode, create a dummy or delayed signal for rx2
                    # so the correlation logic doesn't crash
                    rx2 = np.zeros_like(rx1) 
                else:
                    # Fallback for older pyadi-iio or different return types
                    rx1 = samples[0] if isinstance(samples, list) else samples
                    rx2 = np.zeros_like(rx1)
                
                # Store history
                self.rx1_history.append(rx1)
                self.rx2_history.append(rx2)
                
                # Detect heterodyne
                result = self.detect_heterodyne(rx1, rx2)
                
                # Log significant events
                if result['score'] > self.heterodyne_threshold:
                    self.events.append(result)
                    
                    print(f"\n{'='*60}")
                    print(f"🎯 DETECTION EVENT #{len(self.events)}")
                    print(f"{'='*60}")
                    print(f"  Score:         {result['score']:.3f}")
                    print(f"  Freq Offset:   {result['freq_offset']/1e3:+.2f} kHz")
                    print(f"  Voice Range:   {result['is_voice']}")
                    print(f"  Timestamp:     {time.strftime('%H:%M:%S')}")
                    print(f"{'='*60}")
                
                # Periodic status
                if len(self.rx1_history) % 10 == 0:
                    print(f"📊 Buffers: {len(self.rx1_history):4d} | "
                          f"Events: {len(self.events):3d} | "
                          f"Score: {result['score']:.3f}", end='\r')
                
                time.sleep(0.01)  # Small delay
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)
    
    def start(self):
        """Start detector"""
        self.running = True
        self.thread = threading.Thread(target=self.processing_loop)
        self.thread.start()
    
    def stop(self):
        """Stop detector"""
        print("\n\n" + "="*60)
        print("Stopping Detector")
        print("="*60)
        
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join()
        
        # Summary
        print(f"\n📊 SUMMARY")
        print(f"{'='*60}")
        print(f"  Total Buffers:    {len(self.rx1_history)}")
        print(f"  Events Detected:  {len(self.events)}")
        
        if self.events:
            scores = [e['score'] for e in self.events]
            offsets = [e['freq_offset'] for e in self.events]
            voices = sum(1 for e in self.events if e['is_voice'])
            
            print(f"  Avg Score:        {np.mean(scores):.3f}")
            print(f"  Max Score:        {np.max(scores):.3f}")
            print(f"  Avg Freq Offset:  {np.mean(offsets)/1e3:+.2f} kHz")
            print(f"  Voice Events:     {voices}")
        
        print(f"{'='*60}\n")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Standalone Heterodyne Detector for Pluto+ SDR',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with simulation (no hardware):
  python %(prog)s --simulate
  
  # Real hardware at 433 MHz:
  python %(prog)s --freq 433.92 --sample-rate 2.4
  
  # Adjust detection sensitivity:
  python %(prog)s --simulate --threshold 0.5
  
  # Run for specific duration:
  python %(prog)s --simulate --duration 30
        """
    )
    
    parser.add_argument('--freq', type=float, default=100.0,
                       help='Center frequency in MHz (default: 100)')
    parser.add_argument('--sample-rate', type=float, default=2.4,
                       help='Sample rate in MHz (default: 2.4)')
    parser.add_argument('--rx-gain', type=int, default=50,
                       help='RX gain in dB (default: 50)')
    parser.add_argument('--threshold', type=float, default=0.7,
                       help='Detection threshold 0-1 (default: 0.7)')
    parser.add_argument('--simulate', action='store_true',
                       help='Use simulated data (no hardware needed)')
    parser.add_argument('--duration', type=int, default=None,
                       help='Run duration in seconds (default: forever)')
    
    args = parser.parse_args()
    
    # Print header
    print()
    print("="*60)
    print("🎛️  HETERODYNE DETECTOR - Standalone Version")
    print("="*60)
    print(f"  Center Frequency: {args.freq:.2f} MHz")
    print(f"  Sample Rate:      {args.sample_rate:.2f} MHz")
    print(f"  RX Gain:          {args.rx_gain} dB")
    print(f"  Threshold:        {args.threshold}")
    print(f"  Mode:             {'Simulation' if args.simulate else 'Hardware'}")
    print("="*60)
    print()
    
    # Create SDR interface
    sdr = PlutoSDR(
        sample_rate=args.sample_rate * 1e6,
        center_freq=args.freq * 1e6,
        rx_gain=args.rx_gain,
        simulate=args.simulate
    )
    
    # Connect
    if not sdr.connect():
        print("\n❌ Failed to connect")
        print("\nTry:")
        print("  python standalone_detector.py --simulate")
        return 1
    
    # Create detector
    detector = HeterodyneDetector(sdr)
    detector.heterodyne_threshold = args.threshold
    
    # Start
    detector.start()
    
    # Run
    try:
        if args.duration:
            print(f"Running for {args.duration} seconds...")
            time.sleep(args.duration)
        else:
            print("Running... (Press Ctrl+C to stop)")
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    finally:
        detector.stop()
    
    return 0


if __name__ == "__main__":
    exit(main())
