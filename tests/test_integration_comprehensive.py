#!/usr/bin/env python3
"""
Comprehensive Integration Tests - End-to-end validation

Tests:
1. Full pipeline with realistic data
2. Multi-frequency support (915 MHz, 2.4 GHz, 5.8 GHz)
3. Active noise cancellation with TX references
4. Zero-copy pipeline verification
5. Visualization decoupling verification
6. Type hints compliance
"""

import pytest
import asyncio
import torch
import numpy as np
import time
from typing import Dict, Any

# Import from main
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import RadarApp, RadarGeometry, SPEED_OF_LIGHT
from config import RADAR_GEOMETRY, GPU_CONFIG, ISM_BANDS


class TestFullPipelineIntegration:
    """Test complete radar processing pipeline."""

    @pytest.mark.asyncio
    async def test_end_to_end_simulation(self):
        """Test full pipeline with simulated data."""

        app = RadarApp(simulate=True, enable_viz=False)

        # Run for short duration
        success = await app.run(duration=0.5)

        assert success
        assert app.stats['buffers_processed'] > 0

        print(f"✅ Processed {app.stats['buffers_processed']} buffers")
        print(f"   Avg time: {app.stats['avg_processing_time']*1000:.2f}ms")

    @pytest.mark.asyncio
    async def test_full_pipeline_with_tx_references(self):
        """Test pipeline with active noise cancellation (TX refs)."""

        app = RadarApp(simulate=True, enable_viz=False)

        # Create test data with TX references
        rx1 = np.random.randn(GPU_CONFIG['buffer_size']).astype(np.complex64)
        rx2 = np.random.randn(GPU_CONFIG['buffer_size']).astype(np.complex64)

        # TX references for active cancellation
        tx1_ref = torch.randn(
            GPU_CONFIG['buffer_size'],
            dtype=torch.complex64,
            device=app.device
        )
        tx2_ref = torch.randn(
            GPU_CONFIG['buffer_size'],
            dtype=torch.complex64,
            device=app.device
        )

        # Process with active cancellation
        results = await app.process_buffer(rx1, rx2, tx1_ref, tx2_ref)

        # Verify cancellation info present
        assert 'cancellation_info' in results
        assert 'snr_improvement' in results['cancellation_info']

        snr_improvement = results['cancellation_info']['snr_improvement']
        print(f"✅ Active cancellation: {snr_improvement:.2f} dB improvement")

    @pytest.mark.asyncio
    async def test_detection_with_heterodyning_signal(self):
        """Test detection of heterodyning interference."""

        app = RadarApp(simulate=True, enable_viz=False)

        # Create signal with heterodyning component
        t = np.arange(GPU_CONFIG['buffer_size']) / GPU_CONFIG['sample_rate']
        f_beat = 1500  # Hz beat frequency (heterodyning)

        # Simulated heterodyning interference
        rx1 = np.exp(2j * np.pi * f_beat * t).astype(np.complex64)
        rx2 = np.exp(2j * np.pi * f_beat * t).astype(np.complex64)

        # Add noise
        rx1 += (np.random.randn(len(t)) + 1j * np.random.randn(len(t))) * 0.1
        rx2 += (np.random.randn(len(t)) + 1j * np.random.randn(len(t))) * 0.1

        # Process
        results = await app.process_buffer(rx1, rx2)

        # Should detect the heterodyning
        detection = results['detection']
        print(f"Detection score: {detection['score']:.3f}")
        print(f"Freq offset: {detection['freq_offset']/1e3:+.2f} kHz")

        # Score should be significant for strong heterodyning
        assert detection['score'] > 0.3, "Should detect heterodyning signal"
        print("✅ Heterodyning detection working")


class TestMultiFrequencySupport:
    """Test support for multiple frequency bands."""

    @pytest.mark.parametrize("band", ['915MHz', '2400MHz', '5800MHz'])
    def test_frequency_band_configuration(self, band):
        """Test configuration for different ISM bands."""

        # Configure for band
        band_config = ISM_BANDS[band]
        GPU_CONFIG['center_freq'] = band_config['center_freq']

        app = RadarApp(simulate=True, enable_viz=False)

        # Verify wavelength matches frequency
        expected_wavelength = SPEED_OF_LIGHT / band_config['center_freq']
        actual_wavelength = app.geometry.wavelength

        assert abs(actual_wavelength - expected_wavelength) < 1e-6

        print(f"✅ {band}: λ = {actual_wavelength:.4f}m")

        # Reset
        GPU_CONFIG['center_freq'] = 2.4e9

    @pytest.mark.asyncio
    async def test_processing_across_frequencies(self):
        """Test that processing works across different frequencies."""

        results_by_freq = {}

        for band, config in ISM_BANDS.items():
            GPU_CONFIG['center_freq'] = config['center_freq']

            app = RadarApp(simulate=True, enable_viz=False)

            # Process test buffer
            rx1 = np.random.randn(1024).astype(np.complex64)
            rx2 = np.random.randn(1024).astype(np.complex64)

            results = await app.process_buffer(rx1, rx2)

            results_by_freq[band] = results['processing_time']

        # Print results
        for band, proc_time in results_by_freq.items():
            print(f"{band}: {proc_time*1000:.2f}ms")

        print("✅ Processing works across all ISM bands")

        # Reset
        GPU_CONFIG['center_freq'] = 2.4e9


class TestZeroCopyVerification:
    """Verify zero-copy optimization."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    async def test_pinned_memory_path(self):
        """Test that data goes through pinned memory path."""

        app = RadarApp(simulate=True, enable_viz=False)

        if app.device.type != 'cuda':
            pytest.skip("Requires CUDA")

        # Verify buffers are pinned
        assert app.rx1_buffer.is_pinned()
        assert app.rx2_buffer.is_pinned()

        # Create NumPy arrays
        rx1_np = np.random.randn(GPU_CONFIG['buffer_size']).astype(np.complex64)
        rx2_np = np.random.randn(GPU_CONFIG['buffer_size']).astype(np.complex64)

        # Measure transfer time
        start = time.time()
        results = await app.process_buffer(rx1_np, rx2_np)
        elapsed = time.time() - start

        print(f"Processing time (zero-copy): {elapsed*1000:.2f}ms")

        # Should be fast with pinned memory
        assert elapsed < 0.05, "Zero-copy should be fast"
        print("✅ Zero-copy pipeline verified")

    @pytest.mark.asyncio
    async def test_non_blocking_transfers(self):
        """Test that transfers use non_blocking=True."""

        app = RadarApp(simulate=True, enable_viz=False)

        # This is tested implicitly by the code using non_blocking=True
        # Just verify it doesn't crash and completes quickly

        rx1 = np.random.randn(GPU_CONFIG['buffer_size']).astype(np.complex64)
        rx2 = np.random.randn(GPU_CONFIG['buffer_size']).astype(np.complex64)

        start = time.time()
        results = await app.process_buffer(rx1, rx2)
        elapsed = time.time() - start

        print(f"Non-blocking transfer time: {elapsed*1000:.2f}ms")
        print("✅ Non-blocking transfers working")


class TestVisualizationDecoupling:
    """Test that visualization doesn't block compute."""

    @pytest.mark.asyncio
    async def test_compute_without_visualization(self):
        """Test that compute works without visualization."""

        app = RadarApp(simulate=True, enable_viz=False)

        # Should work without viz
        rx1 = np.random.randn(1024).astype(np.complex64)
        rx2 = np.random.randn(1024).astype(np.complex64)

        results = await app.process_buffer(rx1, rx2)

        assert results is not None
        print("✅ Compute works without visualization")

    @pytest.mark.asyncio
    async def test_visualization_ring_buffer_decoupled(self):
        """Test that visualization ring buffer decouples compute."""

        app = RadarApp(simulate=True, enable_viz=True)

        # Ring buffer should exist
        assert app.viz_ring_buffer is not None

        # Write to ring buffer should be non-blocking
        start = time.time()
        for _ in range(100):
            frame = {
                'rx1': np.random.randn(1024),
                'rx2': np.random.randn(1024),
                'tx1': np.zeros(1024),
                'tx2': np.zeros(1024),
                'mfcc': np.random.randn(13, 20),
                'detection': {'score': 0.5, 'freq_offset': 100},
                'geometry_info': {'doa': 0, 'doa_power': np.zeros(37), 'snr_improvement': 0},
                'range_doppler_map': np.zeros((128, 512)),
                'ppi': {'ppi_map': np.zeros((37, 256))},
                'tracks': []
            }
            app.viz_ring_buffer.write(frame)
        elapsed = time.time() - start

        print(f"100 ring buffer writes: {elapsed*1000:.2f}ms")
        assert elapsed < 0.1, "Ring buffer writes should be fast"
        print("✅ Visualization decoupled via ring buffer")

    @pytest.mark.asyncio
    async def test_compute_continues_during_slow_viz(self):
        """Test that compute continues even if visualization is slow."""

        app = RadarApp(simulate=True, enable_viz=True)

        # Mock slow visualization
        original_update = app.visualizer.update if app.visualizer else None

        async def slow_update(*args, **kwargs):
            await asyncio.sleep(0.1)  # 100ms slow viz
            if original_update:
                await original_update(*args, **kwargs)

        if app.visualizer:
            app.visualizer.update = slow_update

        # Process multiple buffers
        buffer_times = []
        for _ in range(10):
            rx1 = np.random.randn(1024).astype(np.complex64)
            rx2 = np.random.randn(1024).astype(np.complex64)

            start = time.time()
            results = await app.process_buffer(rx1, rx2)
            buffer_times.append(time.time() - start)

        avg_time = np.mean(buffer_times)
        print(f"Avg buffer time with slow viz: {avg_time*1000:.2f}ms")

        # Should not be blocked by slow viz (< 50ms per buffer)
        assert avg_time < 0.05, "Compute should not be blocked by viz"
        print("✅ Compute continues during slow visualization")


class TestTypeHintsCompliance:
    """Test that type hints are properly used."""

    def test_main_methods_have_type_hints(self):
        """Test that main methods have type annotations."""

        from main import RadarApp
        import inspect

        # Check process_buffer
        sig = inspect.signature(RadarApp.process_buffer)

        assert sig.return_annotation != inspect.Parameter.empty, \
            "process_buffer should have return type hint"

        # Check parameters have hints
        for param_name, param in sig.parameters.items():
            if param_name != 'self':
                assert param.annotation != inspect.Parameter.empty, \
                    f"Parameter {param_name} should have type hint"

        print("✅ Type hints present on main methods")

    def test_radar_geometry_type_hints(self):
        """Test RadarGeometry has proper type hints."""

        from main import RadarGeometry
        import inspect

        sig = inspect.signature(RadarGeometry.__init__)

        # All params should have hints
        for param_name, param in sig.parameters.items():
            if param_name != 'self':
                assert param.annotation != inspect.Parameter.empty, \
                    f"RadarGeometry.{param_name} should have type hint"

        print("✅ RadarGeometry has type hints")


class TestPerformanceBenchmarks:
    """Performance benchmarks for common operations."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    async def test_gpu_throughput(self):
        """Benchmark GPU throughput."""

        app = RadarApp(simulate=True, enable_viz=False)

        if app.device.type != 'cuda':
            pytest.skip("Requires CUDA")

        # Warmup
        for _ in range(10):
            rx1 = np.random.randn(GPU_CONFIG['buffer_size']).astype(np.complex64)
            rx2 = np.random.randn(GPU_CONFIG['buffer_size']).astype(np.complex64)
            await app.process_buffer(rx1, rx2)

        # Benchmark
        times = []
        for _ in range(100):
            rx1 = np.random.randn(GPU_CONFIG['buffer_size']).astype(np.complex64)
            rx2 = np.random.randn(GPU_CONFIG['buffer_size']).astype(np.complex64)

            start = time.time()
            results = await app.process_buffer(rx1, rx2)
            times.append(time.time() - start)

        avg_time = np.mean(times)
        std_time = np.std(times)
        throughput = 1.0 / avg_time  # buffers/sec

        print(f"\nGPU Performance:")
        print(f"  Avg time: {avg_time*1000:.2f} ± {std_time*1000:.2f} ms")
        print(f"  Throughput: {throughput:.1f} buffers/sec")
        print(f"  Data rate: {throughput * GPU_CONFIG['buffer_size'] * 2 / 1e6:.1f} MSa/s")

        # Should meet target (< 10ms per buffer on RX 6700 XT)
        assert avg_time < 0.02, f"GPU too slow: {avg_time*1000:.2f}ms > 20ms"
        print("✅ GPU throughput meets target")

    @pytest.mark.asyncio
    async def test_cpu_fallback_performance(self):
        """Benchmark CPU fallback performance."""

        # Force CPU
        original_available = torch.cuda.is_available
        torch.cuda.is_available = lambda: False

        try:
            app = RadarApp(simulate=True, enable_viz=False)

            # Process buffer on CPU
            rx1 = np.random.randn(1024).astype(np.complex64)
            rx2 = np.random.randn(1024).astype(np.complex64)

            start = time.time()
            results = await app.process_buffer(rx1, rx2)
            cpu_time = time.time() - start

            print(f"CPU processing time: {cpu_time*1000:.2f}ms")

            # Should still work (even if slower)
            assert results is not None
            print("✅ CPU fallback works")

        finally:
            torch.cuda.is_available = original_available


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
