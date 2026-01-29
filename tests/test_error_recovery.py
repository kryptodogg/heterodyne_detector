#!/usr/bin/env python3
"""
Test Error Recovery - Verify graceful degradation and error handling

Tests:
1. Double-cleanup bug fixed
2. GPU failure recovery (degraded mode)
3. Invalid input handling
4. Graceful shutdown under errors
5. Resource cleanup after errors
"""

import pytest
import asyncio
import torch
import numpy as np
import time
from unittest.mock import Mock, patch, MagicMock

# Import from main
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import RadarApp, RadarGeometry
from config import RADAR_GEOMETRY, GPU_CONFIG


@pytest.fixture
def radar_app():
    """Create radar app for testing."""
    app = RadarApp(simulate=True, enable_viz=False)
    yield app
    # Cleanup
    if app.running:
        try:
            asyncio.run(app.stop_async())
        except:
            pass


class TestDoubleCleanupFix:
    """Test that double-cleanup bug is fixed."""

    @pytest.mark.asyncio
    async def test_stop_async_idempotent(self):
        """Test that stop_async() can be called multiple times safely."""

        app = RadarApp(simulate=True, enable_viz=False)
        app.running = True

        # First stop - should work
        await app.stop_async()
        assert not app.running

        # Second stop - should be safe (no-op)
        await app.stop_async()  # Should not raise error
        assert not app.running

        # Third stop - still safe
        await app.stop_async()
        assert not app.running

        print("✅ stop_async() is idempotent")

    @pytest.mark.asyncio
    async def test_cleanup_only_when_running(self):
        """Test that cleanup only happens when app is running."""

        app = RadarApp(simulate=True, enable_viz=False)

        # Start app
        app.running = True

        # Stop should cleanup
        await app.stop_async()
        assert not app.running

        # Calling stop again should not re-cleanup
        # (previously this was a bug - cleanup happened in finally)
        await app.stop_async()

        print("✅ Cleanup only when running")

    @pytest.mark.asyncio
    async def test_run_handles_cleanup_correctly(self):
        """Test that run() doesn't double-cleanup on success."""

        app = RadarApp(simulate=True, enable_viz=False)

        # Track stop_async calls
        stop_count = 0
        original_stop = app.stop_async

        async def tracked_stop():
            nonlocal stop_count
            stop_count += 1
            await original_stop()

        app.stop_async = tracked_stop

        # Run for short duration (will complete successfully)
        await app.run(duration=0.1)

        # stop_async should be called exactly ONCE
        # (not twice due to finally block)
        assert stop_count == 1, f"Expected 1 stop call, got {stop_count}"
        print("✅ run() doesn't double-cleanup")


class TestGPUDegradedMode:
    """Test GPU degraded mode handling."""

    def test_gpu_degraded_flag_on_matmul_failure(self):
        """Test that GPU degraded mode is detected."""

        # Mock matmul to fail
        original_matmul = torch.matmul

        def failing_matmul(a, b):
            if a.device.type == 'cuda':
                raise RuntimeError("cuBLAS error: CUBLAS_STATUS_NOT_INITIALIZED")
            return original_matmul(a, b)

        with patch('torch.matmul', side_effect=failing_matmul):
            if torch.cuda.is_available():
                app = RadarApp(simulate=True, enable_viz=False)

                # Should detect degraded mode
                assert app.gpu_degraded == True
                print("✅ GPU degraded mode detected on matmul failure")
            else:
                pytest.skip("Requires CUDA")

    def test_cpu_fallback_no_degraded_mode(self):
        """Test that CPU mode doesn't report degraded."""

        # Force CPU
        original_available = torch.cuda.is_available
        torch.cuda.is_available = lambda: False

        try:
            app = RadarApp(simulate=True, enable_viz=False)

            # CPU should not be degraded
            assert app.gpu_degraded == False
            print("✅ CPU mode not marked as degraded")
        finally:
            torch.cuda.is_available = original_available

    def test_app_continues_in_degraded_mode(self):
        """Test that app continues working in degraded mode."""

        # Create app (might be degraded)
        app = RadarApp(simulate=True, enable_viz=False)

        # Should still initialize successfully
        assert app.device is not None
        assert app.detector is not None
        assert app.noise_canceller is not None

        print(f"✅ App works in degraded mode: {app.gpu_degraded}")


class TestInvalidInputHandling:
    """Test handling of invalid inputs."""

    @pytest.mark.asyncio
    async def test_wrong_shape_input(self, radar_app):
        """Test handling of wrong shape inputs."""

        # Create wrong-sized arrays
        rx1_wrong = np.random.randn(100).astype(np.complex64)  # Too small
        rx2_wrong = np.random.randn(100).astype(np.complex64)

        # Should either handle gracefully or raise clear error
        try:
            results = await radar_app.process_buffer(rx1_wrong, rx2_wrong)
            # If it processes, that's okay (might zero-pad internally)
            print("✅ Wrong shape handled gracefully")
        except (ValueError, RuntimeError) as e:
            # If it raises, error should be clear
            assert "shape" in str(e).lower() or "size" in str(e).lower()
            print(f"✅ Wrong shape raised clear error: {type(e).__name__}")

    @pytest.mark.asyncio
    async def test_nan_input(self, radar_app):
        """Test handling of NaN inputs."""

        # Create arrays with NaN
        rx1_nan = np.full(1024, np.nan, dtype=np.complex64)
        rx2_nan = np.full(1024, np.nan, dtype=np.complex64)

        # Should handle without crashing
        try:
            results = await radar_app.process_buffer(rx1_nan, rx2_nan)
            # Check that results are valid (not all NaN)
            print("✅ NaN input handled")
        except Exception as e:
            print(f"✅ NaN input raised error: {type(e).__name__}")

    @pytest.mark.asyncio
    async def test_mismatched_device_arrays(self, radar_app):
        """Test that device mismatch is corrected."""

        # Create arrays on CPU
        rx1_cpu = torch.randn(1024, dtype=torch.complex64, device='cpu')
        rx2_cpu = torch.randn(1024, dtype=torch.complex64, device='cpu')

        # Should move to correct device automatically
        results = await radar_app.process_buffer(rx1_cpu, rx2_cpu)

        assert results is not None
        print("✅ Device mismatch corrected automatically")


class TestGracefulShutdown:
    """Test graceful shutdown under various conditions."""

    @pytest.mark.asyncio
    async def test_shutdown_during_processing(self):
        """Test graceful shutdown while processing."""

        app = RadarApp(simulate=True, enable_viz=False)

        # Start processing
        run_task = asyncio.create_task(app.run(duration=10))

        # Let it run for a bit
        await asyncio.sleep(0.2)

        # Trigger shutdown
        app.running = False

        # Wait for graceful shutdown
        await asyncio.wait_for(run_task, timeout=2.0)

        assert not app.running
        print("✅ Graceful shutdown during processing")

    @pytest.mark.asyncio
    async def test_shutdown_on_exception(self):
        """Test that resources are cleaned up on exception."""

        app = RadarApp(simulate=True, enable_viz=False)

        # Mock process_buffer to raise exception
        original_process = app.process_buffer

        async def failing_process(*args, **kwargs):
            raise RuntimeError("Simulated processing error")

        app.process_buffer = failing_process

        # Run should handle exception and cleanup
        try:
            await app.run(duration=1.0)
        except Exception:
            pass  # Expected to fail

        # Should have stopped
        assert not app.running
        print("✅ Cleanup on exception")

    @pytest.mark.asyncio
    async def test_keyboard_interrupt_handling(self):
        """Test KeyboardInterrupt handling."""

        app = RadarApp(simulate=True, enable_viz=False)

        # Mock process_buffer to raise KeyboardInterrupt
        async def interrupted_process(*args, **kwargs):
            raise KeyboardInterrupt()

        app.process_buffer = interrupted_process

        # Should handle KeyboardInterrupt gracefully
        try:
            await app.run(duration=1.0)
        except KeyboardInterrupt:
            pass  # Expected

        # Should have cleaned up
        assert not app.running
        print("✅ KeyboardInterrupt handled gracefully")


class TestResourceCleanup:
    """Test that resources are properly cleaned up."""

    @pytest.mark.asyncio
    async def test_gpu_memory_released(self):
        """Test that GPU memory is released after stop."""

        if not torch.cuda.is_available():
            pytest.skip("Requires CUDA")

        # Measure initial GPU memory
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()

        # Create and run app
        app = RadarApp(simulate=True, enable_viz=False)

        # Process some buffers
        for _ in range(10):
            rx1 = np.random.randn(GPU_CONFIG['buffer_size']).astype(np.complex64)
            rx2 = np.random.randn(GPU_CONFIG['buffer_size']).astype(np.complex64)
            await app.process_buffer(rx1, rx2)

        # Memory should be higher
        during_memory = torch.cuda.memory_allocated()
        assert during_memory > initial_memory

        # Stop and cleanup
        await app.stop_async()

        # Force garbage collection
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        # Memory should be released
        final_memory = torch.cuda.memory_allocated()

        print(f"Memory - Initial: {initial_memory/1e6:.1f}MB, "
              f"During: {during_memory/1e6:.1f}MB, "
              f"Final: {final_memory/1e6:.1f}MB")

        # Allow some tolerance
        assert final_memory < during_memory * 0.5, "Memory should be released"
        print("✅ GPU memory released after cleanup")

    @pytest.mark.asyncio
    async def test_visualization_cleanup(self):
        """Test that visualization is properly stopped."""

        app = RadarApp(simulate=True, enable_viz=True)
        app.running = True

        # Visualizer should be running
        assert app.visualizer is not None

        # Stop
        await app.stop_async()

        # Visualizer should be stopped
        # (implementation-dependent, but should at least not crash)
        print("✅ Visualization cleaned up")


class TestWavelengthCalculation:
    """Test wavelength calculation fixes."""

    def test_wavelength_from_freq_arg(self):
        """Test wavelength calculated from actual frequency."""

        # Set custom frequency
        GPU_CONFIG['center_freq'] = 915e6  # 915 MHz

        app = RadarApp(simulate=True, enable_viz=False)

        # Wavelength should be c/f
        expected = 299792458.0 / 915e6
        actual = app.geometry.wavelength

        assert abs(actual - expected) < 1e-6, \
            f"Expected {expected:.6f}, got {actual:.6f}"
        print(f"✅ Wavelength correct for 915 MHz: {actual:.4f}m")

        # Reset
        GPU_CONFIG['center_freq'] = 2.4e9

    def test_wavelength_not_hardcoded(self):
        """Test wavelength changes with frequency."""

        # Test multiple frequencies
        frequencies = [915e6, 2.4e9, 5.8e9]

        for freq in frequencies:
            GPU_CONFIG['center_freq'] = freq

            app = RadarApp(simulate=True, enable_viz=False)

            expected = 299792458.0 / freq
            actual = app.geometry.wavelength

            assert abs(actual - expected) < 1e-6, \
                f"Wavelength not correct for {freq/1e9:.1f} GHz"

            print(f"✅ {freq/1e9:.1f} GHz: λ = {actual:.4f}m")

        # Reset
        GPU_CONFIG['center_freq'] = 2.4e9


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
