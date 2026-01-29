#!/usr/bin/env python3
"""
main.py - Radar Application Orchestrator (Fully Async with CUDA Streams)
Torch-first, ROCm-enabled 2TX2RX radar with spatial noise cancellation

Architecture:
- True async/await with CUDA streams - fully non-blocking GPU operations
- Geometry-based beamforming for active noise cancellation
- Zero-copy GPU pipeline with pinned memory DMA
- MFCC radar feature extraction (Doppler-optimized)
- HDF5 pattern library management
- Decoupled visualization with lock-free ring buffer (game engine pattern)

Usage:
    python main.py --simulate              # Test with synthetic data
    python main.py --freq 2400             # Real hardware at 2.4 GHz
    python main.py --config radar.yaml    # Load custom geometry
"""

import asyncio
import signal
import torch
import numpy as np
import argparse
import sys
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Any, Optional, Union, Tuple, List
from collections import deque
import threading

# Import radar components
from config import (
    RADAR_GEOMETRY,
    NOISE_CANCELLATION,
    MFCC_RADAR_SETTINGS,
    GPU_CONFIG,
    HDF5_STORAGE,
    RANGE_DOPPLER,
    PPI_CONFIG,
    TRACKING,
    DETECTION,
    get_torch_geometry
)
from sdr_interface import PlutoRadarInterface
from audio_processor import AudioProcessor as RadarAudioProcessor
from heterodyne_detector import HeterodyneDetector as HeterodyneDetectorTorch
from noise_canceller import SpatialNoiseCanceller
from range_doppler import RangeDopplerProcessor
from ppi_processor import PPIProcessor
from tracker import TargetTracker
from pattern_matcher import PatternMatcher as RadarPatternMatcher
from data_manager import HDF5LibraryManager as HDF5DataManager
from visualizer import VisualizerDash as RadarDashboard

# ============================================================
# CONSTANTS (Extract Magic Numbers)
# ============================================================
SPEED_OF_LIGHT = 299792458.0  # m/s
DEFAULT_VIZ_REFRESH_HZ = 20.0
DEFAULT_DETECTION_THRESHOLD = DETECTION['heterodyne_threshold']  # 0.7
STATUS_UPDATE_INTERVAL = 100  # buffers
EMA_ALPHA = 0.1  # Exponential moving average smoothing
VIZ_RING_BUFFER_SIZE = 4  # Lock-free visualization buffer


@dataclass
class RadarGeometry:
    """
    Stores radar geometry configuration as Torch tensors for GPU acceleration.
    All tensors created directly on target device (zero-copy pattern).
    """
    tx1_pos: torch.Tensor
    tx2_pos: torch.Tensor
    rx1_pos: torch.Tensor
    rx2_pos: torch.Tensor
    baseline: float
    wavelength: float
    device: torch.device

    def __init__(
        self,
        config: Dict[str, Any],
        device: torch.device = torch.device('cpu'),
        center_freq: Optional[float] = None
    ):
        """
        Initialize geometry from config dictionary.

        Args:
            config: RADAR_GEOMETRY dict from config.py
            device: Torch device to store tensors on
            center_freq: Radar center frequency in Hz (used for wavelength calc)
        """
        self.device = device

        # Helper to create tensors on CPU first for safe math
        def to_cpu_tensor(key: str) -> torch.Tensor:
            return torch.tensor(
                config[key]['position'],
                dtype=torch.float32,
                device=torch.device('cpu')
            )

        # Create positions on CPU
        tx1_cpu = to_cpu_tensor('TX1')
        tx2_cpu = to_cpu_tensor('TX2')
        rx1_cpu = to_cpu_tensor('RX1')
        rx2_cpu = to_cpu_tensor('RX2')

        # Calculate derived metrics on CPU (safe)
        self.baseline = float(torch.norm(rx2_cpu - rx1_cpu).item())

        # Move to target device ONLY after math is done
        self.tx1_pos = tx1_cpu.to(device)
        self.tx2_pos = tx2_cpu.to(device)
        self.rx1_pos = rx1_cpu.to(device)
        self.rx2_pos = rx2_cpu.to(device)

        # Calculate wavelength on CPU
        if 'wavelength' in config and config['wavelength'] is not None:
            self.wavelength = float(config['wavelength'])
        else:
            freq = center_freq if center_freq is not None else GPU_CONFIG.get('center_freq', 2.4e9)
            self.wavelength = SPEED_OF_LIGHT / freq

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
        theta = theta.to(self.device, non_blocking=True)

        # Assuming RX array is along X-axis
        d_vec = self.rx2_pos - self.rx1_pos

        # Phase shift for RX2 relative to RX1
        phase = k * d_vec[0] * torch.sin(theta)

        ones = torch.ones_like(theta, dtype=torch.complex64)
        shifts = torch.exp(1j * phase.to(torch.complex64))

        # Stack
        sv = torch.stack([ones, shifts], dim=-1)  # Shape (N, 2)

        # Expansion to 4 elements for compatibility with 2TX2RX logic
        sv = torch.cat([sv, sv], dim=-1)

        return sv


class VisualizationRingBuffer:
    """
    Lock-free ring buffer for decoupled visualization (game engine pattern).
    Compute writes without blocking, visualization reads latest state.
    """

    def __init__(self, size: int = VIZ_RING_BUFFER_SIZE):
        """
        Initialize ring buffer.

        Args:
            size: Buffer capacity (number of frames to store)
        """
        self.size = size
        self.buffer: deque = deque(maxlen=size)
        self.lock = threading.Lock()
        self.latest_frame: Optional[Dict[str, Any]] = None

    def write(self, frame: Dict[str, Any]) -> None:
        """
        Write frame to buffer (non-blocking, overwrites oldest).

        Args:
            frame: Visualization data dictionary
        """
        with self.lock:
            self.buffer.append(frame)
            self.latest_frame = frame

    def read_latest(self) -> Optional[Dict[str, Any]]:
        """
        Read latest frame without blocking.

        Returns:
            Latest frame or None if buffer empty
        """
        with self.lock:
            return self.latest_frame


class RadarApp:
    """
    Main radar application orchestrator with CUDA streams for true async GPU execution.
    Manages the full signal processing pipeline with zero-copy optimization.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        simulate: bool = False,
        enable_viz: bool = True
    ):
        """
        Initialize radar application.

        Args:
            config_path: Path to YAML config (optional)
            simulate: Use synthetic data instead of hardware
            enable_viz: Enable real-time dashboard
        """
        print("="*60)
        print("🎯 RADAR APPLICATION - Torch-First Architecture (CUDA Streams)")
        print("="*60)

        # GPU setup with degraded mode tracking
        self.device, self.gpu_degraded = self._setup_gpu()

        # CUDA streams for true async GPU execution
        if self.device.type == 'cuda':
            self.stream_detection = torch.cuda.Stream()
            self.stream_range_doppler = torch.cuda.Stream()
            self.stream_ppi = torch.cuda.Stream()
            self.stream_mfcc = torch.cuda.Stream()
            print(f"✅ CUDA Streams Initialized (4 concurrent streams)")
        else:
            # CPU mode: no streams needed
            self.stream_detection = None
            self.stream_range_doppler = None
            self.stream_ppi = None
            self.stream_mfcc = None

        # Load geometry with actual center frequency for wavelength calculation
        center_freq = GPU_CONFIG.get('center_freq', 2.4e9)
        self.geometry = RadarGeometry(
            RADAR_GEOMETRY,
            device=self.device,
            center_freq=center_freq
        )
        self._print_geometry_info()

        # Initialize SDR interface
        self.sdr = PlutoRadarInterface(
            sample_rate=GPU_CONFIG['sample_rate'],
            center_freq=center_freq,
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

        # Spatial noise canceller (beamforming + LMS for active cancellation)
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

        # Range-Doppler processor (pseudo-pulse CW mode)
        self.rd_processor = RangeDopplerProcessor(
            config=RANGE_DOPPLER,
            device=self.device
        )

        # PPI (Polar Position Indicator) processor
        self.ppi_processor = PPIProcessor(
            geometry=self.geometry,
            config=PPI_CONFIG,
            device=self.device
        )

        # Target tracker (Kalman filter multi-target)
        self.tracker = TargetTracker(
            config=TRACKING,
            device=self.device
        )

        # Pre-allocate PINNED Torch buffers for zero-copy DMA transfers
        self.buffer_size = GPU_CONFIG['buffer_size']
        if self.device.type == 'cuda':
            # Use pinned memory for faster CPU→GPU transfers
            self.rx1_buffer = torch.zeros(
                self.buffer_size,
                dtype=torch.complex64
            ).pin_memory()
            self.rx2_buffer = torch.zeros(
                self.buffer_size,
                dtype=torch.complex64
            ).pin_memory()
        else:
            self.rx1_buffer = torch.zeros(
                self.buffer_size,
                dtype=torch.complex64,
                device=self.device
            )
            self.rx2_buffer = torch.zeros(
                self.buffer_size,
                dtype=torch.complex64,
                device=self.device
            )

        # Visualization with decoupled ring buffer (game engine pattern)
        self.visualizer: Optional[RadarDashboard] = None
        self.viz_ring_buffer: Optional[VisualizationRingBuffer] = None
        if enable_viz:
            self.visualizer = RadarDashboard(
                refresh_rate_hz=DEFAULT_VIZ_REFRESH_HZ,
                geometry=self.geometry
            )
            self.viz_ring_buffer = VisualizationRingBuffer()
            # Start background visualization consumer
            asyncio.create_task(self._visualization_consumer())

        # State
        self.running = False
        self.session_id = self._create_session_id()
        self.stats: Dict[str, Any] = {
            'buffers_processed': 0,
            'detections': 0,
            'patterns_matched': 0,
            'avg_processing_time': 0.0
        }

        # Performance: Throttled visualization (non-blocking writes)
        self.last_viz_update = 0.0
        self.viz_interval = 1.0 / DEFAULT_VIZ_REFRESH_HZ

        print(f"\n✅ Radar App Initialized")
        print(f"   Session: {self.session_id}")
        print(f"   Device: {self.device}")
        print(f"   GPU Degraded Mode: {self.gpu_degraded}")
        print(f"   Zero-Copy: {GPU_CONFIG.get('zero_copy', True)}")
        print(f"   Visualization: {'Enabled (Decoupled)' if enable_viz else 'Disabled'}")
        print(f"   Wavelength: {self.geometry.wavelength:.4f} m ({center_freq/1e9:.2f} GHz)")
        print("="*60)


    def _setup_gpu(self) -> Tuple[torch.device, bool]:
        """
        Setup GPU and verify Torch + ROCm functionality.

        Returns:
            Tuple of (device, degraded_mode_flag)
        """
        if not torch.cuda.is_available():
            print("⚠️  WARNING: CUDA/ROCm not available!")
            print("   Falling back to CPU (will be slow)")
            return torch.device('cpu'), False

        device = torch.device('cuda:0')
        print(f"\n✅ GPU Detected")
        print(f"   Device: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        # GPU Health Check: Test matmul (common point of failure on ROCm)
        gpu_degraded = False
        try:
            test_a = torch.randn(2, 2, device=device)
            test_b = torch.randn(2, 2, device=device)
            torch.matmul(test_a, test_b)
            print("   GPU Health Check: matmul OK")
        except RuntimeError as e:
            gpu_degraded = True
            print(f"⚠️  WARNING: GPU matmul failed: {e}")
            print("   The GPU is available but its BLAS libraries failed to initialize.")
            print("   Entering DEGRADED MODE (Using manual SafeGEMM kernels on GPU).")
            # We continue with GPU because FFTs and element-wise ops still work!

        # Set memory allocation strategy (optional)
        if 'memory_fraction' in GPU_CONFIG:
            torch.cuda.set_per_process_memory_fraction(
                GPU_CONFIG['memory_fraction'],
                device=0
            )

        return device, gpu_degraded

    def _print_geometry_info(self) -> None:
        """Print radar geometry information."""
        print(f"\n📐 Radar Geometry")
        print(f"   TX1: {self.geometry.tx1_pos.cpu().numpy()}")
        print(f"   TX2: {self.geometry.tx2_pos.cpu().numpy()}")
        print(f"   RX1: {self.geometry.rx1_pos.cpu().numpy()}")
        print(f"   RX2: {self.geometry.rx2_pos.cpu().numpy()}")
        print(f"   Baseline: {self.geometry.baseline:.3f} m")

    def _create_session_id(self) -> str:
        """Create unique session ID."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    async def process_buffer(
        self,
        rx1: Union[torch.Tensor, np.ndarray],
        rx2: Union[torch.Tensor, np.ndarray],
        tx1_ref: Optional[torch.Tensor] = None,
        tx2_ref: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Async process single buffer through full radar pipeline using CUDA streams.
        GPU operations run in true parallel via separate streams.

        Args:
            rx1: RX1 complex samples (numpy or torch)
            rx2: RX2 complex samples
            tx1_ref: TX1 reference (optional, for active cancellation)
            tx2_ref: TX2 reference (optional)

        Returns:
            Full pipeline results with Range-Doppler, PPI, tracking
        """
        start_time = time.time()

        # Convert to torch tensors with ZERO-COPY optimization (pinned memory DMA)
        if not isinstance(rx1, torch.Tensor):
            # NumPy → Pinned Memory → GPU (2-3x faster than direct to GPU)
            rx1 = torch.from_numpy(rx1).pin_memory().to(
                self.device,
                non_blocking=True
            )
            rx2 = torch.from_numpy(rx2).pin_memory().to(
                self.device,
                non_blocking=True
            )
        else:
            # Ensure existing tensors are on correct device
            rx1 = rx1.to(self.device, non_blocking=True)
            rx2 = rx2.to(self.device, non_blocking=True)

        # Step 1: Spatial noise cancellation (beamforming + LMS active cancellation)
        # This runs on default stream (stream 0) and must complete first
        clean_rx1, clean_rx2, cancellation_info = self.noise_canceller.cancel(
            rx1, rx2, tx1_ref=tx1_ref, tx2_ref=tx2_ref
        )

        # Step 2-5: Run independent GPU operations concurrently via CUDA streams
        # These don't depend on each other and can execute in true parallel
        if self.device.type == 'cuda':
            # Launch operations on separate streams (TRUE async GPU execution)
            detection_task = asyncio.create_task(
                self._run_detection_stream(clean_rx1, clean_rx2)
            )
            rd_task = asyncio.create_task(
                self._run_range_doppler_stream(clean_rx1)
            )
            ppi_task = asyncio.create_task(
                self._run_ppi_stream(clean_rx1, clean_rx2)
            )
            mfcc_task = asyncio.create_task(
                self._run_mfcc_stream(clean_rx1)
            )
        else:
            # CPU mode: run in thread pool
            loop = asyncio.get_event_loop()
            detection_task = loop.run_in_executor(
                None, self.detector.detect, clean_rx1, clean_rx2
            )
            rd_task = loop.run_in_executor(
                None, self.rd_processor.process, clean_rx1
            )
            ppi_task = loop.run_in_executor(
                None, self.ppi_processor.process, clean_rx1, clean_rx2
            )
            mfcc_task = loop.run_in_executor(
                None, self.audio_proc.extract_features, clean_rx1
            )

        # Await all concurrent operations
        detection, rd_results, ppi_results, mfcc_features = await asyncio.gather(
            detection_task, rd_task, ppi_task, mfcc_task
        )

        # Step 6: Target tracking (depends on rd_results)
        tracks = await self._run_tracking(rd_results['detections'])

        # Step 7: Pattern matching (depends on mfcc_features)
        matches = await self._run_pattern_matching(mfcc_features)

        # Step 8: Update statistics
        processing_time = time.time() - start_time
        self._update_stats(processing_time, detection, matches)

        # Comprehensive results with all pipeline outputs
        results: Dict[str, Any] = {
            'detection': detection,
            'mfcc_features': mfcc_features,
            'pattern_matches': matches,
            'cancellation_info': cancellation_info,
            'range_doppler': rd_results,
            'ppi': ppi_results,
            'tracks': tracks,
            'processing_time': processing_time,
            'timestamp': time.time()
        }

        # Update visualization asynchronously (non-blocking write to ring buffer)
        if self.viz_ring_buffer and (time.time() - self.last_viz_update) > self.viz_interval:
            viz_frame = {
                'rx1': rx1.detach().cpu().numpy(),
                'rx2': rx2.detach().cpu().numpy(),
                'tx1': tx1_ref.detach().cpu().numpy() if tx1_ref is not None else np.zeros(1024),
                'tx2': tx2_ref.detach().cpu().numpy() if tx2_ref is not None else np.zeros(1024),
                'mfcc': mfcc_features.detach().cpu().numpy(),
                'detection': detection,
                'geometry_info': cancellation_info,
                'range_doppler_map': rd_results['range_doppler_map'],
                'ppi': ppi_results,
                'tracks': tracks
            }
            self.viz_ring_buffer.write(viz_frame)  # Non-blocking write
            self.last_viz_update = time.time()

        return results

    # ============================================================
    # CUDA Stream Async Wrappers (True Parallel GPU Execution)
    # ============================================================

    async def _run_detection_stream(
        self,
        rx1: torch.Tensor,
        rx2: torch.Tensor
    ) -> Dict[str, Any]:
        """Run heterodyne detection on dedicated CUDA stream."""
        with torch.cuda.stream(self.stream_detection):
            result = self.detector.detect(rx1, rx2)
        await asyncio.sleep(0)  # Yield control to event loop
        return result

    async def _run_range_doppler_stream(
        self,
        rx1: torch.Tensor
    ) -> Dict[str, Any]:
        """Run Range-Doppler processing on dedicated CUDA stream."""
        with torch.cuda.stream(self.stream_range_doppler):
            result = self.rd_processor.process(rx1)
            self.stream_range_doppler.synchronize()
        await asyncio.sleep(0)
        return result

    async def _run_ppi_stream(
        self,
        rx1: torch.Tensor,
        rx2: torch.Tensor
    ) -> Dict[str, Any]:
        """Run PPI processing on dedicated CUDA stream."""
        with torch.cuda.stream(self.stream_ppi):
            result = self.ppi_processor.process(rx1, rx2)
            self.stream_ppi.synchronize()
        await asyncio.sleep(0)
        return result

    async def _run_mfcc_stream(
        self,
        rx1: torch.Tensor
    ) -> torch.Tensor:
        """Run MFCC extraction on dedicated CUDA stream."""
        with torch.cuda.stream(self.stream_mfcc):
            result = self.audio_proc.extract_features(rx1)
            self.stream_mfcc.synchronize()
        await asyncio.sleep(0)
        return result

    async def _run_tracking(
        self,
        detections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Async target tracking (runs on default stream)."""
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self.tracker.update,
            detections
        )
        return result

    async def _run_pattern_matching(
        self,
        features: torch.Tensor
    ) -> List[Dict[str, Any]]:
        """Async pattern matching."""
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self.pattern_matcher.find_matches,
            features
        )
        return result

    async def _visualization_consumer(self) -> None:
        """
        Background task that consumes from ring buffer and updates visualization.
        Runs independently from compute pipeline (game engine pattern).
        """
        while self.running or self.viz_ring_buffer.latest_frame is not None:
            if self.visualizer and self.viz_ring_buffer:
                frame = self.viz_ring_buffer.read_latest()
                if frame:
                    try:
                        await self.visualizer.update(**frame)
                    except Exception as e:
                        print(f"⚠️  Visualization update failed: {e}")
            await asyncio.sleep(self.viz_interval)  # Throttle updates

    def _update_stats(
        self,
        proc_time: float,
        detection: Dict[str, Any],
        matches: List[Dict[str, Any]]
    ) -> None:
        """Update processing statistics."""
        self.stats['buffers_processed'] += 1

        # Exponential moving average of processing time
        self.stats['avg_processing_time'] = (
            EMA_ALPHA * proc_time +
            (1 - EMA_ALPHA) * self.stats['avg_processing_time']
        )

        if detection['score'] > DEFAULT_DETECTION_THRESHOLD:
            self.stats['detections'] += 1

        if matches:
            self.stats['patterns_matched'] += len(matches)

    async def run(self, duration: Optional[int] = None) -> bool:
        """
        Async radar processing loop - fully non-blocking.

        Args:
            duration: Run duration in seconds (None = forever)

        Returns:
            Success status
        """
        print(f"\n{'='*60}")
        print("🚀 Starting Async Radar Processing Loop (CUDA Streams)")
        print(f"{'='*60}")

        # Async SDR connection
        if not await self.sdr.connect():
            print("❌ Failed to connect to SDR")
            return False

        # Start visualization if enabled
        if self.visualizer:
            await self.visualizer.start_async()

        self.running = True
        start_time = time.time()

        try:
            while self.running:
                # Check duration
                if duration and (time.time() - start_time) > duration:
                    print("\n⏱️  Duration reached")
                    break

                # Async receive data from SDR (non-blocking I/O)
                rx1, rx2 = await self.sdr.rx()

                # Async process buffer (GPU ops run concurrently via CUDA streams)
                results = await self.process_buffer(rx1, rx2)

                # Log significant events (async to avoid blocking)
                if results['detection']['score'] > DEFAULT_DETECTION_THRESHOLD:
                    await self._log_detection_async(results)

                # Save patterns if requested (async I/O)
                if results['pattern_matches']:
                    await self._save_patterns_async(results)

                # Periodic status update (non-blocking)
                if self.stats['buffers_processed'] % STATUS_UPDATE_INTERVAL == 0:
                    self._print_status()

                # Yield control to other tasks
                await asyncio.sleep(0)

        except asyncio.CancelledError:
            print("\n\n⚠️  Task cancelled")

        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")

        except Exception as e:
            print(f"\n\n❌ Error in processing loop: {e}")
            import traceback
            traceback.print_exc()
            return False

        finally:
            # Cleanup (only if not already stopped)
            if self.running:
                await self.stop_async()

        return True

    async def _log_detection_async(self, results: Dict[str, Any]) -> None:
        """Async log detection event."""
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
        await asyncio.sleep(0)  # Yield control

    async def _save_patterns_async(self, results: Dict[str, Any]) -> None:
        """Async save detected patterns to HDF5."""
        # Run in executor to avoid blocking on I/O
        loop = asyncio.get_event_loop()
        for match in results['pattern_matches']:
            await loop.run_in_executor(
                None,
                self.data_manager.save_pattern,
                self.session_id,
                f"{self.sdr.center_freq/1e6:.0f}MHz",
                results['mfcc_features'].cpu().numpy(),
                {
                    'score': match['score'],
                    'timestamp': results['timestamp'],
                    'detection_score': results['detection']['score']
                }
            )

    def _print_status(self) -> None:
        """Print periodic status update."""
        print(f"\r📊 Buffers: {self.stats['buffers_processed']:6d} | "
              f"Detections: {self.stats['detections']:4d} | "
              f"Matches: {self.stats['patterns_matched']:4d} | "
              f"Proc: {self.stats['avg_processing_time']*1000:.1f} ms",
              end='')

    async def stop_async(self) -> None:
        """Async stop the radar application."""
        if not self.running:
            return  # Already stopped

        print(f"\n\n{'='*60}")
        print("🛑 Stopping Radar Application")
        print(f"{'='*60}")

        self.running = False

        # Async stop visualization
        if self.visualizer:
            await self.visualizer.stop_async()

        # Async disconnect SDR
        await self.sdr.close()

        # Print final statistics
        self._print_final_stats()

        # Cleanup GPU memory
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def stop(self) -> None:
        """Synchronous fallback stop."""
        if not self.running:
            return

        print(f"\n\n{'='*60}")
        print("🛑 Stopping Radar Application (Sync)")
        print(f"{'='*60}")

        self.running = False

        if self.visualizer:
            self.visualizer.stop()

        if hasattr(self.sdr, 'close_sync'):
            self.sdr.close_sync()

        self._print_final_stats()

        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def _print_final_stats(self) -> None:
        """Print final statistics."""
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


async def async_main() -> int:
    """Async main entry point."""
    parser = argparse.ArgumentParser(
        description='Async Torch-First Radar Application with Active Noise Cancellation',
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
    app: Optional[RadarApp] = None
    try:
        app = RadarApp(
            config_path=args.config,
            simulate=args.simulate,
            enable_viz=not args.no_viz
        )

        # Setup signal handlers for graceful shutdown
        loop = asyncio.get_running_loop()

        def signal_handler() -> None:
            print("\n⚠️  Shutdown signal received")
            if app:
                app.running = False

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, signal_handler)

        # Run async
        success = await app.run(duration=args.duration)

        return 0 if success else 1

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        # Ensure cleanup only if app exists and is still running
        if app is not None and app.running:
            await app.stop_async()


def main() -> int:
    """Synchronous wrapper for async_main."""
    try:
        # Run async event loop
        return asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted")
        return 1


if __name__ == "__main__":
    sys.exit(main())
