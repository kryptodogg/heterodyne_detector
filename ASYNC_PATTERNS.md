# Async Patterns Quick Reference

## Core Pattern: Everything Awaits Data

```python
# ❌ WRONG - Blocks event loop
def process():
    data = blocking_io_call()
    result = blocking_computation(data)
    return result

# ✅ RIGHT - Async non-blocking
async def process():
    data = await async_io_call()
    result = await async_computation(data)
    return result
```

## Pattern 1: I/O Operations (SDR, Files, Network)

**Use `run_in_executor()` for blocking I/O**:

```python
async def rx(self):
    loop = asyncio.get_event_loop()
    # Blocking hardware call runs in thread pool
    samples = await loop.run_in_executor(None, self._rx_hardware)
    return samples

def _rx_hardware(self):
    """Sync method called in executor"""
    return self.sdr.rx()  # Blocking hardware read
```

## Pattern 2: Fast GPU Operations

**Yield control after fast operations**:

```python
async def _run_detection(self, rx1, rx2):
    # GPU op completes in microseconds
    result = self.detector.detect(rx1, rx2)

    # Yield to allow other tasks to run
    await asyncio.sleep(0)

    return result
```

**Why**: GPU ops are async at hardware level, but synchronous in Python. `await asyncio.sleep(0)` yields control without delay.

## Pattern 3: Concurrent Independent Operations

**Use `asyncio.gather()` for parallelism**:

```python
async def process_all(self, data):
    # These don't depend on each other - run concurrently
    results = await asyncio.gather(
        operation_a(data),
        operation_b(data),
        operation_c(data),
        operation_d(data)
    )

    return results  # All 4 completed in parallel
```

**Speedup**: 4 ops × 100ms each = 400ms sequential → ~100ms concurrent (4x faster)

## Pattern 4: Sequential Dependencies

**await each step when outputs depend on inputs**:

```python
async def pipeline(self, raw_data):
    # Step 1: Must complete first
    cleaned = await clean_data(raw_data)

    # Step 2: Depends on cleaned data, but these 3 are independent
    feature_a, feature_b, feature_c = await asyncio.gather(
        extract_feature_a(cleaned),
        extract_feature_b(cleaned),
        extract_feature_c(cleaned)
    )

    # Step 3: Depends on features
    result = await combine_features(feature_a, feature_b, feature_c)

    return result
```

**Visualization**:
```
Time →
├─ clean_data ────────┐
                       ├─ extract_a ─┐
                       ├─ extract_b ─┤ (concurrent)
                       ├─ extract_c ─┘
                                      ├─ combine ─→ result
```

## Pattern 5: Event Loop Control

**Main async loop pattern**:

```python
async def run(self):
    while self.running:
        # Get data (async)
        data = await self.source.get()

        # Process (async)
        result = await self.process(data)

        # Output (async)
        await self.output.put(result)

        # Yield control (critical!)
        await asyncio.sleep(0)
```

**Why the final `await asyncio.sleep(0)`**: Prevents tight loop from starving other tasks.

## Pattern 6: Signal Handling

**Graceful shutdown**:

```python
async def main():
    app = MyApp()

    # Register signal handlers
    loop = asyncio.get_running_loop()

    def shutdown():
        print("Shutting down...")
        app.running = False

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown)

    try:
        await app.run()
    finally:
        await app.cleanup()
```

**Result**: Ctrl+C triggers graceful async shutdown, not abrupt termination.

## Pattern 7: Resource Cleanup

**Use try/finally for async resources**:

```python
async def run(self):
    try:
        await self.connect()
        await self.process_loop()
    finally:
        await self.disconnect()
        await self.cleanup()
```

## Pattern 8: Timeout Protection

**Prevent operations from hanging**:

```python
async def get_data_with_timeout(self):
    try:
        data = await asyncio.wait_for(
            self.sdr.rx(),
            timeout=5.0  # 5 second timeout
        )
        return data
    except asyncio.TimeoutError:
        print("Operation timed out")
        return None
```

## Pattern 9: Background Tasks

**Run tasks concurrently with main loop**:

```python
async def main():
    # Start background task
    monitor_task = asyncio.create_task(monitor_health())

    try:
        # Main processing
        await process_loop()
    finally:
        # Cancel background task
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass
```

## Pattern 10: Async Lock (When Needed)

**For shared state across tasks**:

```python
class DataBuffer:
    def __init__(self):
        self._lock = asyncio.Lock()
        self._data = []

    async def append(self, item):
        async with self._lock:
            self._data.append(item)

    async def get_all(self):
        async with self._lock:
            return self._data.copy()
```

**Note**: Use sparingly - locks reduce concurrency. Prefer message passing via `asyncio.Queue`.

## Common Mistakes

### ❌ Mistake 1: Forgetting await

```python
# WRONG - Creates coroutine object, doesn't execute
async def bad():
    result = async_operation()  # Missing await!
    return result

# RIGHT
async def good():
    result = await async_operation()
    return result
```

### ❌ Mistake 2: Blocking in Async Function

```python
# WRONG - Blocks event loop
async def bad():
    time.sleep(1)  # Blocking!
    return "done"

# RIGHT
async def good():
    await asyncio.sleep(1)  # Non-blocking
    return "done"
```

### ❌ Mistake 3: Not Yielding in Tight Loop

```python
# WRONG - Starves event loop
async def bad():
    while True:
        data = process_immediately()  # Never yields
        store(data)

# RIGHT
async def good():
    while True:
        data = process_immediately()
        store(data)
        await asyncio.sleep(0)  # Yield control
```

### ❌ Mistake 4: Sequential When Concurrent Possible

```python
# WRONG - Sequential (slow)
async def bad():
    a = await task_a()
    b = await task_b()
    c = await task_c()
    return a, b, c

# RIGHT - Concurrent (fast)
async def good():
    results = await asyncio.gather(task_a(), task_b(), task_c())
    return results
```

## Debugging Async Code

### Enable asyncio debug mode:

```python
# In main entry point
import asyncio

asyncio.run(main(), debug=True)
```

Shows warnings for:
- Coroutines that were never awaited
- Tasks that take too long
- Blocking calls in the event loop

### Detect blocking operations:

```bash
# Run with warnings
python -W default main.py
```

### Profiling async code:

```python
import time

async def profile():
    start = time.time()
    result = await operation()
    elapsed = time.time() - start
    print(f"Operation took {elapsed*1000:.1f}ms")
    return result
```

## Performance Tips

1. **Use gather() for true parallelism**
   - `asyncio.gather()` runs tasks concurrently
   - Sequential awaits run one after another

2. **Yield often in CPU-bound loops**
   - Add `await asyncio.sleep(0)` every N iterations
   - Prevents event loop starvation

3. **Run blocking ops in executor**
   - File I/O: `await loop.run_in_executor(None, file.read)`
   - CPU-bound: `await loop.run_in_executor(pool, heavy_compute)`

4. **Keep locks short**
   - Acquire lock, update state, release immediately
   - Don't hold locks across await points

5. **Use queues for producer/consumer**
   - `asyncio.Queue` is async-safe
   - Better than locks for data passing

## Radar System Application

Our radar system uses these patterns:

```python
async def run(self):
    while self.running:
        # I/O pattern (SDR)
        rx1, rx2 = await self.sdr.rx()  # Executor

        # Concurrent processing pattern
        detection, rd, ppi, mfcc = await asyncio.gather(
            self._run_detection(rx1, rx2),
            self._run_range_doppler(rx1),
            self._run_ppi(rx1, rx2),
            self._run_mfcc(rx1)
        )

        # Fast GPU op + yield pattern
        async def _run_detection(self, rx1, rx2):
            result = self.detector.detect(rx1, rx2)
            await asyncio.sleep(0)
            return result

        # Loop control pattern
        await asyncio.sleep(0)
```

**Result**: 4x speedup, fully non-blocking, responsive to signals.
