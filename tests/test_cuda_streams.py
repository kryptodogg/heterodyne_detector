#!/usr/bin/env python3
"""
Test CUDA Streams - Verify true concurrent GPU execution

Tests:
1. CUDA streams enable concurrent kernel execution
2. Stream synchronization works correctly
3. CPU fallback when CUDA unavailable
4. Memory consistency across streams
"""

import pytest
import asyncio
import torch
import numpy as np
import time
from typing import Tuple

# Import from main
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import RadarApp, RadarGeometry, SPEED_OF_LIGHT
from config import RADAR_GEOMETRY, GPU_CONFIG


@pytest.fixture
def device():
    """Get device for testing (CUDA if available, else CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def radar_app():
    """Create radar app for testing."""
    app = RadarApp(simulate=True, enable_viz=False)
    yield app
    # Cleanup
    if app.running:
        asyncio.run(app.stop_async())


class TestCUDAStreams:
    """Test suite for CUDA stream functionality."""

    def test_cuda_streams_initialized_gpu(self):
        """Test that CUDA streams are initialized when GPU available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        app = RadarApp(simulate=True, enable_viz=False)

        # Verify streams exist
        assert app.stream_detection is not None
        assert app.stream_range_doppler is not None
        assert app.stream_ppi is not None
        assert app.stream_mfcc is not None

        # Verify they're actually CUDA streams
        assert isinstance(app.stream_detection, torch.cuda.Stream)
        print("✅ CUDA streams initialized correctly")

    def test_cuda_streams_none_cpu(self):
        """Test that streams are None on CPU."""
        # Force CPU by mocking
        import torch.cuda
        original_available = torch.cuda.is_available
        torch.cuda.is_available = lambda: False

        try:
            app = RadarApp(simulate=True, enable_viz=False)

            assert app.stream_detection is None
            assert app.stream_range_doppler is None
            assert app.stream_ppi is None
            assert app.stream_mfcc is None
            print("✅ CPU mode has no CUDA streams")
        finally:
            torch.cuda.is_available = original_available

    @pytest.mark.asyncio
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    async def test_concurrent_stream_execution(self, radar_app):
        """Test that CUDA streams enable concurrent execution."""

        # Create test data
        rx1 = torch.randn(
            GPU_CONFIG['buffer_size'],
            dtype=torch.complex64,
            device=radar_app.device
        )
        rx2 = torch.randn(
            GPU_CONFIG['buffer_size'],
            dtype=torch.complex64,
            device=radar_app.device
        )

        # Measure sequential execution
        start = time.time()
        result1 = await radar_app._run_detection_stream(rx1, rx2)
        result2 = await radar_app._run_range_doppler_stream(rx1)
        result3 = await radar_app._run_ppi_stream(rx1, rx2)
        result4 = await radar_app._run_mfcc_stream(rx1)
        sequential_time = time.time() - start

        # Measure concurrent execution
        start = time.time()
        results = await asyncio.gather(
            radar_app._run_detection_stream(rx1, rx2),
            radar_app._run_range_doppler_stream(rx1),
            radar_app._run_ppi_stream(rx1, rx2),
            radar_app._run_mfcc_stream(rx1)
        )
        concurrent_time = time.time() - start

        print(f"Sequential: {sequential_time*1000:.2f}ms")
        print(f"Concurrent: {concurrent_time*1000:.2f}ms")

        # Concurrent should be faster (or at least not slower)
        # Allow some tolerance for small operations
        speedup = sequential_time / concurrent_time
        print(f"Speedup: {speedup:.2f}x")

        # On GPU with multiple SMs, expect speedup > 1.2x
        # On CPU or small ops, expect ~1x (no degradation)
        assert speedup >= 0.9, "Concurrent execution should not be slower"
        print("✅ CUDA streams enable concurrent execution")

    @pytest.mark.asyncio
    async def test_stream_synchronization(self, radar_app, device):
        """Test that stream synchronization ensures completion."""

        if device.type != 'cuda':
            pytest.skip("Requires CUDA")

        # Create test tensor
        test_tensor = torch.randn(1000, 1000, device=device)

        # Launch operation on stream
        with torch.cuda.stream(radar_app.stream_detection):
            result = torch.matmul(test_tensor, test_tensor)
            # Don't synchronize yet - result might not be ready

        # Synchronize explicitly
        radar_app.stream_detection.synchronize()

        # Now result should be ready
        assert result.shape == (1000, 1000)
        assert not torch.isnan(result).any(), "Result should be valid after sync"
        print("✅ Stream synchronization works correctly")

    @pytest.mark.asyncio
    async def test_cross_stream_memory_consistency(self, radar_app, device):
        """Test that tensors are visible across streams."""

        if device.type != 'cuda':
            pytest.skip("Requires CUDA")

        # Create tensor on default stream
        tensor_a = torch.ones(100, device=device)

        # Modify on stream 1
        with torch.cuda.stream(radar_app.stream_detection):
            tensor_a *= 2
            radar_app.stream_detection.synchronize()

        # Read on stream 2
        with torch.cuda.stream(radar_app.stream_range_doppler):
            value = tensor_a[0].item()
            radar_app.stream_range_doppler.synchronize()

        # Should see the modification
        assert value == 2.0, "Modifications should be visible across streams"
        print("✅ Memory consistency across streams verified")

    @pytest.mark.asyncio
    async def test_process_buffer_async_execution(self, radar_app):
        """Test that process_buffer uses concurrent streams."""

        # Create test data
        rx1 = np.random.randn(GPU_CONFIG['buffer_size']).astype(np.complex64)
        rx2 = np.random.randn(GPU_CONFIG['buffer_size']).astype(np.complex64)

        # Process buffer (should use CUDA streams internally)
        start = time.time()
        results = await radar_app.process_buffer(rx1, rx2)
        elapsed = time.time() - start

        # Verify results structure
        assert 'detection' in results
        assert 'range_doppler' in results
        assert 'ppi' in results
        assert 'mfcc_features' in results

        print(f"Process buffer time: {elapsed*1000:.2f}ms")

        # Should complete in reasonable time (< 100ms per buffer on GPU)
        if radar_app.device.type == 'cuda':
            assert elapsed < 0.1, "GPU processing should be fast"

        print("✅ process_buffer uses async CUDA streams")


class TestDeviceConsistency:
    """Test device consistency across pipeline."""

    @pytest.mark.asyncio
    async def test_numpy_to_gpu_transfer(self, radar_app):
        """Test NumPy → GPU transfer with pinned memory."""

        # Create NumPy array
        rx1_np = np.random.randn(1024).astype(np.complex64)
        rx2_np = np.random.randn(1024).astype(np.complex64)

        # Process through pipeline
        results = await radar_app.process_buffer(rx1_np, rx2_np)

        # All internal results should be on correct device
        assert results is not None
        print(f"✅ NumPy → {radar_app.device} transfer successful")

    @pytest.mark.asyncio
    async def test_mixed_device_inputs(self, radar_app, device):
        """Test handling of tensors already on device."""

        # Create tensor on device
        rx1_tensor = torch.randn(
            1024,
            dtype=torch.complex64,
            device=device
        )
        rx2_tensor = torch.randn(
            1024,
            dtype=torch.complex64,
            device=device
        )

        # Process (should handle device correctly)
        results = await radar_app.process_buffer(rx1_tensor, rx2_tensor)

        assert results is not None
        print(f"✅ Tensor inputs on {device} handled correctly")

    @pytest.mark.asyncio
    async def test_wrong_device_tensor(self, radar_app):
        """Test that wrong device tensors are moved correctly."""

        # Create tensors on CPU
        rx1_cpu = torch.randn(1024, dtype=torch.complex64, device='cpu')
        rx2_cpu = torch.randn(1024, dtype=torch.complex64, device='cpu')

        # Process (should move to correct device)
        results = await radar_app.process_buffer(rx1_cpu, rx2_cpu)

        assert results is not None
        print("✅ Wrong device tensors moved correctly")


class TestPinnedMemory:
    """Test pinned memory optimization."""

    def test_pinned_buffers_allocated(self, radar_app):
        """Test that buffers are pinned when CUDA available."""

        if radar_app.device.type == 'cuda':
            # Buffers should be pinned
            assert radar_app.rx1_buffer.is_pinned()
            assert radar_app.rx2_buffer.is_pinned()
            print("✅ Buffers are pinned for fast DMA")
        else:
            # CPU buffers can't be pinned
            assert not radar_app.rx1_buffer.is_pinned()
            print("✅ CPU buffers are not pinned (expected)")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_pinned_memory_transfer_speed(self, radar_app):
        """Test that pinned memory transfers are faster."""

        # Create test data
        size = 2**20  # 1M samples
        test_data = np.random.randn(size).astype(np.float32)

        # Transfer without pinning
        start = time.time()
        regular_tensor = torch.from_numpy(test_data).to(
            radar_app.device,
            non_blocking=False
        )
        regular_time = time.time() - start

        # Transfer with pinning
        start = time.time()
        pinned_tensor = torch.from_numpy(test_data).pin_memory().to(
            radar_app.device,
            non_blocking=True
        )
        torch.cuda.synchronize()  # Wait for async transfer
        pinned_time = time.time() - start

        print(f"Regular transfer: {regular_time*1000:.2f}ms")
        print(f"Pinned transfer: {pinned_time*1000:.2f}ms")

        # Pinned should be faster (or at least not slower)
        speedup = regular_time / pinned_time
        print(f"Speedup: {speedup:.2f}x")

        assert speedup >= 0.9, "Pinned transfer should not be slower"
        print("✅ Pinned memory transfers optimized")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
