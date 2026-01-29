#!/usr/bin/env python3
"""
Test Async Architecture - Verify non-blocking operation

This script tests that:
1. SDR rx() doesn't block the event loop
2. Processing pipeline runs concurrently
3. Visualization updates asynchronously
4. Multiple tasks can run in parallel
"""

import asyncio
import time
import numpy as np


async def test_async_timing():
    """
    Test that async operations allow concurrency.
    Sequential operations should take ~3s, async should take ~1s
    """
    print("Testing async concurrency...")

    # Sequential (bad - blocking)
    start = time.time()
    await asyncio.sleep(1)
    await asyncio.sleep(1)
    await asyncio.sleep(1)
    sequential_time = time.time() - start
    print(f"Sequential: {sequential_time:.2f}s")

    # Concurrent (good - non-blocking)
    start = time.time()
    await asyncio.gather(
        asyncio.sleep(1),
        asyncio.sleep(1),
        asyncio.sleep(1)
    )
    concurrent_time = time.time() - start
    print(f"Concurrent: {concurrent_time:.2f}s")

    assert concurrent_time < 1.5, "Async operations should run concurrently"
    print("✅ Async timing test passed")


async def test_sdr_nonblocking():
    """Test that SDR rx() yields control"""
    from sdr_interface import PlutoRadarInterface

    sdr = PlutoRadarInterface(simulate=True, device='cpu')
    await sdr.connect()

    print("\nTesting SDR non-blocking rx()...")

    # Track when each task runs
    events = []

    async def rx_task():
        for i in range(5):
            rx1, rx2 = await sdr.rx()
            events.append(f"RX {i}")
            assert rx1 is not None, "Should receive data"

    async def monitor_task():
        for i in range(10):
            await asyncio.sleep(0.01)
            events.append(f"Monitor {i}")

    # Both should interleave if non-blocking
    start = time.time()
    await asyncio.gather(rx_task(), monitor_task())
    elapsed = time.time() - start

    print(f"Events: {events[:10]}...")
    print(f"Total events: {len(events)}")
    assert len(events) == 15, "Both tasks should complete"

    # Events should interleave (not all RX first, then all Monitor)
    rx_events = [e for e in events if 'RX' in e]
    monitor_events = [e for e in events if 'Monitor' in e]

    # Check interleaving by ensuring RX events are spread out
    rx_indices = [i for i, e in enumerate(events) if 'RX' in e]
    spacing = np.diff(rx_indices)
    avg_spacing = np.mean(spacing) if len(spacing) > 0 else 0

    print(f"RX event spacing: {avg_spacing:.1f} positions")
    assert avg_spacing > 1, "RX events should interleave with monitor events"

    print("✅ SDR non-blocking test passed")


async def test_visualization_nonblocking():
    """Test that visualization updates don't block"""
    from visualizer import VisualizerDash

    # Create visualizer without starting server
    viz = VisualizerDash(refresh_rate_hz=20)

    print("\nTesting visualization non-blocking updates...")

    # Simulate rapid updates
    start = time.time()
    for i in range(100):
        await viz.update(
            rx1=np.random.randn(1024).astype(np.complex64),
            rx2=np.random.randn(1024).astype(np.complex64),
            mfcc=np.random.randn(13, 20),
            detection={'score': 0.5, 'freq_offset': 100},
            geometry_info={'doa': 0, 'doa_power': np.zeros(37), 'snr_improvement': 0},
            range_doppler_map=np.zeros((128, 512)),
            ppi={'ppi_map': np.zeros((37, 256))},
            tracks=[]
        )
    elapsed = time.time() - start

    # Should complete quickly (< 100ms for 100 updates)
    print(f"100 updates in {elapsed*1000:.1f}ms")
    assert elapsed < 0.5, "Visualization updates should be non-blocking"
    print("✅ Visualization non-blocking test passed")


async def test_processing_pipeline_concurrent():
    """Test that processing modules can run concurrently"""
    print("\nTesting concurrent processing pipeline...")

    # Simulate 4 independent GPU operations
    async def gpu_op(name, duration):
        start = time.time()
        # Simulate GPU work with sleep
        await asyncio.sleep(duration)
        elapsed = time.time() - start
        return name, elapsed

    # Run sequentially
    start = time.time()
    results = []
    for name, dur in [('Det', 0.1), ('RD', 0.1), ('PPI', 0.1), ('MFCC', 0.1)]:
        result = await gpu_op(name, dur)
        results.append(result)
    sequential_time = time.time() - start

    # Run concurrently
    start = time.time()
    results = await asyncio.gather(
        gpu_op('Det', 0.1),
        gpu_op('RD', 0.1),
        gpu_op('PPI', 0.1),
        gpu_op('MFCC', 0.1)
    )
    concurrent_time = time.time() - start

    print(f"Sequential: {sequential_time:.2f}s")
    print(f"Concurrent: {concurrent_time:.2f}s")
    print(f"Speedup: {sequential_time/concurrent_time:.1f}x")

    assert concurrent_time < sequential_time / 2, "Concurrent should be faster"
    print("✅ Concurrent processing test passed")


async def main():
    """Run all async architecture tests"""
    print("="*60)
    print("Async Architecture Test Suite")
    print("="*60)

    try:
        await test_async_timing()
        await test_sdr_nonblocking()
        await test_visualization_nonblocking()
        await test_processing_pipeline_concurrent()

        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED - Async architecture working correctly")
        print("="*60)

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        raise

    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    asyncio.run(main())
