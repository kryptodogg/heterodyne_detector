#!/usr/bin/env python3
"""
main.py - Radar Application Orchestrator
Torch-first, ROCm-enabled 2TX2RX radar with spatial noise cancellation

Architecture:
- Geometry-based beamforming for noise cancellation
- GPU-accelerated signal processing (Torch)
- MFCC radar feature extraction (Doppler-optimized)
- HDF5 pattern library management
- Real-time Dash visualization (60 Hz target)

Usage:
    python main.py --simulate              # Test with synthetic data
    python main.py --freq 2400             # Real hardware at 2.4 GHz
    python main.py --config radar.yaml    # Load custom geometry
"""

import torch
import numpy as np
import argparse
import sys
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

# Import radar components
from config import (
    RADAR_GEOMETRY,
    NOISE_CANCELLATION,
    MFCC_RADAR_SETTINGS,
    GPU_CONFIG,
    HDF5_STORAGE,
    get_torch_geometry
)
from sdr_interface import PlutoRadarInterface
from audio_processor import AudioProcessor as RadarAudioProcessor
from heterodyne_detector import HeterodyneDetector as HeterodyneDetectorTorch
from noise_canceller import SpatialNoiseCanceller
from pattern_matcher import PatternMatcher as RadarPatternMatcher
from data_manager import HDF5LibraryManager as HDF5DataManager
from visualizer import VisualizerDash as RadarDashboard

@dataclass
class RadarGeometry:
    """
    Stores radar geometry configuration as Torch tensors for GPU acceleration.
    """
    tx1_pos: torch.Tensor
    tx2_pos: torch.Tensor
    rx1_pos: torch.Tensor
    rx2_pos: torch.Tensor
    baseline: float
    wavelength: float

    def __init__(self, config: dict, device: torch.device = torch.device('cpu')):
        """
        Initialize geometry from config dictionary.
        
        Args:
            config: RADAR_GEOMETRY dict from config.py
            device: Torch device to store tensors on
        """
        # Helper to convert numpy/list to tensor
        def to_tensor(key):
            return torch.tensor(config[key]['position'], 
                              dtype=torch.float32, 
                              device=device)

        self.tx1_pos = to_tensor('TX1')
        self.tx2_pos = to_tensor('TX2')
        self.rx1_pos = to_tensor('RX1')
        self.rx2_pos = to_tensor('RX2')
        
        # Calculate derived metrics
        self.baseline = float(torch.norm(self.rx2_pos - self.rx1_pos))
        
        # Calculate wavelength if center_freq is available in global config, 
        # otherwise default to 2.4 GHz logic or config value
        c = 299792458.0
        # If wavelength isn't pre-calculated, approximate it or pass it in. 
        # Here we trust the config or default to 0.125m (2.4GHz)
        self.wavelength = config.get('wavelength') or (c / 2.4e9)
    
    def compute_steering_vector(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Compute steering vectors for given angles.
        
        Args:
            theta: Tensor of angles in radians
            
        Returns:
            Tensor of steering vectors (complex)
        """
        # Basic ULA approximation
        k = 2 * np.pi / self.wavelength
        
        # Ensure theta is on the correct device
        theta = theta.to(self.tx1_pos.device)
        
        # Assuming RX array is along X-axis
        d_vec = self.rx2_pos - self.rx1_pos 
        
        # Phase shift for RX2 relative to RX1
        phase = k * d_vec[0] * torch.sin(theta)
        
        ones = torch.ones_like(theta, dtype=torch.complex64)
        shifts = torch.exp(1j * phase.to(torch.complex64))
        
        # Stack
        sv = torch.stack([ones, shifts], dim=-1) # Shape (N, 2)
        
        # Expansion to 4 elements for compatibility with 2TX2RX logic
        sv = torch.cat([sv, sv], dim=-1)
        
        return sv


class RadarApp:
    """
    Main radar application orchestrator
    Manages the full signal processing pipeline
    """
    
    def __init__(self, config_path=None, simulate=False, enable_viz=True):
        """
        Initialize radar application
        
        Args:
            config_path: Path to YAML config (optional)
            simulate: Use synthetic data instead of hardware
            enable_viz: Enable real-time dashboard
        """
        print("="*60)
        print("🎯 RADAR APPLICATION - Torch-First Architecture")
        print("="*60)
        
        # GPU setup
        self.device = self._setup_gpu()
        
        # Load geometry
        self.geometry = RadarGeometry(RADAR_GEOMETRY)
        self._print_geometry_info()
        
        # Initialize SDR interface
        self.sdr = PlutoRadarInterface(
            sample_rate=GPU_CONFIG['sample_rate'],
            center_freq=GPU_CONFIG['center_freq'],
            simulate=simulate,
            device=self.device
        )
        
        # Audio processor (MFCC radar features)
        self.audio_proc = RadarAudioProcessor(
            **MFCC_RADAR_SETTINGS,
            device=self.device
        )
        
        # Heterodyne detector
        self.detector = HeterodyneDetectorTorch(
            sample_rate=GPU_CONFIG['sample_rate'],
            device=self.device
        )
        
        # Spatial noise canceller (beamforming + LMS)
        self.noise_canceller = SpatialNoiseCanceller(
            geometry=self.geometry,
            config=NOISE_CANCELLATION,
            device=self.device
        )
        
        # Pattern matcher
        self.pattern_matcher = RadarPatternMatcher(
            device=self.device
        )
        
        # Data manager (HDF5)
        self.data_manager = HDF5DataManager(
            base_path=HDF5_STORAGE['base_path']
        )
        
        # Pre-allocate Torch buffers for signal ingress
        self.buffer_size = GPU_CONFIG['buffer_size']
        self.rx1_buffer = torch.zeros(self.buffer_size, dtype=torch.complex64, device=self.device)
        self.rx2_buffer = torch.zeros(self.buffer_size, dtype=torch.complex64, device=self.device)
        
        # Visualization
        self.visualizer = None
        if enable_viz:
            self.visualizer = RadarDashboard(
                refresh_rate_hz=60,
                geometry=self.geometry
            )
        
        # State
        self.running = False
        self.session_id = self._create_session_id()
        self.stats = {
            'buffers_processed': 0,
            'detections': 0,
            'patterns_matched': 0,
            'avg_processing_time': 0.0
        }
        
        print(f"\n✅ Radar App Initialized")
        print(f"   Session: {self.session_id}")
        print(f"   Device: {self.device}")
        print(f"   Visualization: {'Enabled' if enable_viz else 'Disabled'}")
        print("="*60)
    
    def _setup_gpu(self):
        """Setup GPU and verify Torch + ROCm"""
        if not torch.cuda.is_available():
            print("⚠️  WARNING: CUDA/ROCm not available!")
            print("   Falling back to CPU (will be slow)")
            return torch.device('cpu')
        
        device = torch.device('cuda:0')
        print(f"\n✅ GPU Detected")
        print(f"   Device: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Set memory allocation strategy
        torch.cuda.set_per_process_memory_fraction(
            GPU_CONFIG['memory_fraction'], 
            device=0
        )
        
        return device
    
    def _print_geometry_info(self):
        """Print radar geometry information"""
        print(f"\n📐 Radar Geometry")
        print(f"   TX1: {self.geometry.tx1_pos}")
        print(f"   TX2: {self.geometry.tx2_pos}")
        print(f"   RX1: {self.geometry.rx1_pos}")
        print(f"   RX2: {self.geometry.rx2_pos}")
        print(f"   Baseline: {self.geometry.baseline:.3f} m")
    
    def _create_session_id(self):
        """Create unique session ID"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def process_buffer(self, rx1, rx2, tx1_ref=None, tx2_ref=None):
        """
        Process a single buffer of radar data
        
        Args:
            rx1: RX1 complex samples (numpy or torch)
            rx2: RX2 complex samples
            tx1_ref: TX1 reference (optional)
            tx2_ref: TX2 reference (optional)
        
        Returns:
            dict: Processing results
        """
        start_time = time.time()
        
        # Convert to torch tensors
        if not isinstance(rx1, torch.Tensor):
            rx1 = torch.from_numpy(rx1).to(self.device)
            rx2 = torch.from_numpy(rx2).to(self.device)
        
        # Step 1: Spatial noise cancellation (beamforming)
        clean_rx1, clean_rx2, cancellation_info = self.noise_canceller.cancel(
            rx1, rx2
        )
        
        # Step 2: Heterodyne detection
        detection = self.detector.detect(clean_rx1, clean_rx2)
        
        # Step 3: MFCC radar feature extraction
        mfcc_features = self.audio_proc.extract_features(clean_rx1)
        
        # Step 4: Pattern matching
        matches = self.pattern_matcher.find_matches(mfcc_features)
        
        # Step 5: Update statistics
        processing_time = time.time() - start_time
        self._update_stats(processing_time, detection, matches)
        
        # Prepare results
        results = {
            'detection': detection,
            'mfcc_features': mfcc_features,
            'pattern_matches': matches,
            'cancellation_info': cancellation_info,
            'processing_time': processing_time,
            'timestamp': time.time()
        }
        
        # Update visualization if enabled
        if self.visualizer:
            self.visualizer.update(
                rx1=clean_rx1.cpu().numpy(),
                rx2=clean_rx2.cpu().numpy(),
                mfcc=mfcc_features.cpu().numpy(),
                detection=detection,
                geometry_info=cancellation_info
            )
        
        return results
    
    def _update_stats(self, proc_time, detection, matches):
        """Update processing statistics"""
        self.stats['buffers_processed'] += 1
        
        # Running average of processing time
        alpha = 0.1
        self.stats['avg_processing_time'] = (
            alpha * proc_time + 
            (1 - alpha) * self.stats['avg_processing_time']
        )
        
        if detection['score'] > detection.get('threshold', 0.7):
            self.stats['detections'] += 1
        
        if matches:
            self.stats['patterns_matched'] += len(matches)
    
    def run(self, duration=None):
        """
        Run the radar processing loop
        
        Args:
            duration: Run duration in seconds (None = forever)
        """
        print(f"\n{'='*60}")
        print("🚀 Starting Radar Processing Loop")
        print(f"{'='*60}")
        
        if not self.sdr.connect():
            print("❌ Failed to connect to SDR")
            return False
        
        # Start visualization if enabled
        if self.visualizer:
            self.visualizer.start()
        
        self.running = True
        start_time = time.time()
        
        try:
            while self.running:
                # Check duration
                if duration and (time.time() - start_time) > duration:
                    print("\n⏱️  Duration reached")
                    break
                
                # Receive data from SDR
                rx1, rx2 = self.sdr.rx()
                
                # Process buffer
                results = self.process_buffer(rx1, rx2)
                
                # Log significant events
                if results['detection']['score'] > 0.7:
                    self._log_detection(results)
                
                # Save patterns if requested
                if results['pattern_matches']:
                    self._save_patterns(results)
                
                # Periodic status update
                if self.stats['buffers_processed'] % 100 == 0:
                    self._print_status()
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
        
        finally:
            self.stop()
        
        return True
    
    def _log_detection(self, results):
        """Log detection event"""
        det = results['detection']
        print(f"\n{'='*60}")
        print(f"🎯 DETECTION #{self.stats['detections']}")
        print(f"{'='*60}")
        print(f"  Score:       {det['score']:.3f}")
        print(f"  Freq Offset: {det['freq_offset']/1e3:+.2f} kHz")
        print(f"  DOA:         {det.get('doa', 'N/A')}")
        print(f"  Timestamp:   {time.strftime('%H:%M:%S')}")
        
        if results['pattern_matches']:
            print(f"  Matches:     {len(results['pattern_matches'])}")
        
        print(f"{'='*60}")
    
    def _save_patterns(self, results):
        """Save detected patterns to HDF5"""
        for match in results['pattern_matches']:
            self.data_manager.save_pattern(
                session_id=self.session_id,
                band=f"{self.sdr.center_freq/1e6:.0f}MHz",
                features=results['mfcc_features'].cpu().numpy(),
                metadata={
                    'score': match['score'],
                    'timestamp': results['timestamp'],
                    'detection_score': results['detection']['score']
                }
            )
    
    def _print_status(self):
        """Print periodic status update"""
        print(f"\r📊 Buffers: {self.stats['buffers_processed']:6d} | "
              f"Detections: {self.stats['detections']:4d} | "
              f"Matches: {self.stats['patterns_matched']:4d} | "
              f"Proc: {self.stats['avg_processing_time']*1000:.1f} ms",
              end='')
    
    def stop(self):
        """Stop the radar application"""
        print(f"\n\n{'='*60}")
        print("🛑 Stopping Radar Application")
        print(f"{'='*60}")
        
        self.running = False
        
        # Stop visualization
        if self.visualizer:
            self.visualizer.stop()
        
        # Disconnect SDR
        self.sdr.disconnect()
        
        # Print final statistics
        self._print_final_stats()
        
        # Cleanup GPU memory
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
    
    def _print_final_stats(self):
        """Print final statistics"""
        print(f"\n📊 Session Statistics")
        print(f"{'='*60}")
        print(f"  Session ID:       {self.session_id}")
        print(f"  Buffers:          {self.stats['buffers_processed']}")
        print(f"  Detections:       {self.stats['detections']}")
        print(f"  Pattern Matches:  {self.stats['patterns_matched']}")
        print(f"  Avg Proc Time:    {self.stats['avg_processing_time']*1000:.2f} ms")
        
        if self.stats['buffers_processed'] > 0:
            det_rate = self.stats['detections'] / self.stats['buffers_processed'] * 100
            print(f"  Detection Rate:   {det_rate:.2f}%")
        
        print(f"{'='*60}\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Torch-First Radar Application with Spatial Noise Cancellation',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--simulate', action='store_true',
                       help='Use simulated data (no hardware required)')
    parser.add_argument('--freq', type=float, default=2400,
                       help='Center frequency in MHz (default: 2400)')
    parser.add_argument('--sample-rate', type=float, default=10.0,
                       help='Sample rate in MHz (default: 10)')
    parser.add_argument('--duration', type=int, default=None,
                       help='Run duration in seconds (default: forever)')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to YAML config file')
    parser.add_argument('--no-viz', action='store_true',
                       help='Disable real-time visualization')
    parser.add_argument('--save-patterns', action='store_true',
                       help='Save detected patterns to HDF5')
    
    args = parser.parse_args()
    
    # Override config if specified
    if args.freq:
        GPU_CONFIG['center_freq'] = args.freq * 1e6
    if args.sample_rate:
        GPU_CONFIG['sample_rate'] = args.sample_rate * 1e6
    
    # Create radar app
    try:
        app = RadarApp(
            config_path=args.config,
            simulate=args.simulate,
            enable_viz=not args.no_viz
        )
        
        # Run
        success = app.run(duration=args.duration)
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
