# Heterodyne Detector - Test Suite

Comprehensive test suite for the Torch-first GPU-accelerated radar system with CUDA streams.

## Test Organization

### 📁 Test Files

| File | Purpose | Key Tests |
|------|---------|-----------|
| `test_cuda_streams.py` | CUDA stream concurrency and zero-copy | Stream initialization, concurrent execution, pinned memory |
| `test_error_recovery.py` | Error handling and graceful degradation | Double-cleanup fix, GPU degraded mode, invalid inputs |
| `test_memory_leaks.py` | Memory leak detection | GPU memory stability, ring buffer bounds, tensor cleanup |
| `test_integration_comprehensive.py` | End-to-end integration | Full pipeline, multi-frequency, performance benchmarks |

### 🎯 Test Coverage

#### Critical Fixes Tested:
- ✅ **CUDA Streams** - True async GPU execution with concurrent streams
- ✅ **Type Hints** - Comprehensive type annotations validated
- ✅ **Error Handling** - Double-cleanup bug fixed, idempotent stop
- ✅ **Device Consistency** - NumPy/Torch/CPU/GPU device validation
- ✅ **Zero-Copy Pipeline** - Pinned memory DMA transfers
- ✅ **Wavelength Calculation** - Dynamic frequency-based (no hardcoding)
- ✅ **Visualization Decoupling** - Ring buffer game engine pattern
- ✅ **GPU Degraded Mode** - Graceful fallback on BLAS failure

#### Performance Benchmarks:
- GPU throughput (target: < 10ms per buffer)
- Pinned memory transfer speedup (2-3x faster)
- CUDA stream concurrency (2-4x parallelism)
- Memory stability (< 10 MB growth over 1000 iterations)

## Running Tests

### Prerequisites
```bash
pip install pytest pytest-asyncio pytest-cov
```

### Quick Start
```bash
# Run all tests
./run_tests.sh

# Run specific test suite
./run_tests.sh cuda          # CUDA-specific tests
./run_tests.sh errors        # Error recovery tests
./run_tests.sh memory        # Memory leak tests
./run_tests.sh integration   # Integration tests

# Run with coverage
./run_tests.sh coverage
```

### Manual Execution
```bash
# All tests
python -m pytest tests/ -v

# Specific test file
python -m pytest tests/test_cuda_streams.py -v

# Specific test class
python -m pytest tests/test_cuda_streams.py::TestCUDAStreams -v

# Specific test method
python -m pytest tests/test_cuda_streams.py::TestCUDAStreams::test_concurrent_stream_execution -v

# Skip slow tests
python -m pytest tests/ -v -m "not slow"

# With output
python -m pytest tests/ -v -s
```

## Test Categories

### 1. CUDA Streams Tests (`test_cuda_streams.py`)

**Purpose**: Verify true concurrent GPU execution via CUDA streams.

**Key Tests**:
- `test_cuda_streams_initialized_gpu()` - Streams created on CUDA
- `test_cuda_streams_none_cpu()` - No streams on CPU
- `test_concurrent_stream_execution()` - Parallel GPU ops (2-4x speedup)
- `test_stream_synchronization()` - Sync ensures completion
- `test_cross_stream_memory_consistency()` - Tensors visible across streams
- `test_process_buffer_async_execution()` - Full pipeline uses streams

**Device Consistency Tests**:
- `test_numpy_to_gpu_transfer()` - NumPy → GPU with pinned memory
- `test_mixed_device_inputs()` - Handles tensors already on device
- `test_wrong_device_tensor()` - Auto-moves to correct device

**Pinned Memory Tests**:
- `test_pinned_buffers_allocated()` - Buffers pinned on CUDA
- `test_pinned_memory_transfer_speed()` - 2-3x faster DMA

**Example**:
```bash
python -m pytest tests/test_cuda_streams.py -v
```

### 2. Error Recovery Tests (`test_error_recovery.py`)

**Purpose**: Validate graceful error handling and degraded mode.

**Key Tests**:
- `test_stop_async_idempotent()` - Can call stop() multiple times safely
- `test_cleanup_only_when_running()` - No double-cleanup bug
- `test_run_handles_cleanup_correctly()` - Stop called exactly once
- `test_gpu_degraded_flag_on_matmul_failure()` - Detects BLAS failure
- `test_app_continues_in_degraded_mode()` - Works despite degradation
- `test_wrong_shape_input()` - Handles invalid shapes
- `test_nan_input()` - NaN inputs don't crash
- `test_mismatched_device_arrays()` - Auto-corrects device mismatches
- `test_shutdown_during_processing()` - Graceful interrupt
- `test_wavelength_not_hardcoded()` - Dynamic wavelength calc

**Example**:
```bash
python -m pytest tests/test_error_recovery.py::TestDoubleCleanupFix -v
```

### 3. Memory Leak Tests (`test_memory_leaks.py`)

**Purpose**: Detect memory leaks in long-running operations.

**Key Tests**:
- `test_long_run_memory_stable()` - GPU memory stable over 1000 iterations
- `test_tensor_references_released()` - Tensors GC'd properly
- `test_pinned_memory_reuse()` - Buffers reused (not reallocated)
- `test_ring_buffer_bounded_size()` - Viz buffer doesn't grow
- `test_ring_buffer_overwrites_oldest()` - FIFO behavior verified
- `test_numpy_arrays_not_retained()` - NumPy arrays cleaned up
- `test_detach_cpu_numpy_cleanup()` - .detach().cpu().numpy() freed
- `test_streams_dont_leak_memory()` - CUDA streams don't leak

**Example**:
```bash
python -m pytest tests/test_memory_leaks.py::TestGPUMemoryLeaks -v -s
```

### 4. Integration Tests (`test_integration_comprehensive.py`)

**Purpose**: End-to-end validation with realistic scenarios.

**Key Tests**:
- `test_end_to_end_simulation()` - Full pipeline for 0.5s
- `test_full_pipeline_with_tx_references()` - Active noise cancellation
- `test_detection_with_heterodyning_signal()` - Detects interference
- `test_frequency_band_configuration()` - 915 MHz / 2.4 GHz / 5.8 GHz
- `test_processing_across_frequencies()` - Works on all ISM bands
- `test_pinned_memory_path()` - Zero-copy verified
- `test_compute_without_visualization()` - Headless mode works
- `test_visualization_ring_buffer_decoupled()` - Game engine pattern
- `test_main_methods_have_type_hints()` - Type hints compliance
- `test_gpu_throughput()` - Benchmark (target: < 10ms/buffer)

**Example**:
```bash
python -m pytest tests/test_integration_comprehensive.py -v -s
```

## Expected Results

### On GPU (CUDA/ROCm):
```
✅ CUDA streams initialized (4 concurrent streams)
✅ Concurrent execution: 2-4x speedup
✅ Pinned memory: 2-3x faster transfers
✅ GPU throughput: < 10ms per buffer
✅ Memory stable: < 10 MB growth over 1000 iterations
✅ Zero-copy pipeline active
```

### On CPU:
```
✅ No CUDA streams (expected)
✅ Thread pool execution fallback
✅ All functionality preserved (slower)
✅ No GPU degraded mode
```

### On GPU with BLAS Failure (ROCm):
```
⚠️  GPU Degraded Mode detected
✅ App continues with manual kernels
✅ FFTs and element-wise ops still work
✅ Performance degraded but functional
```

## Performance Targets

| Metric | Target | Measured On |
|--------|--------|-------------|
| **Buffer Processing** | < 10 ms | RX 6700 XT |
| **CUDA Stream Speedup** | 2-4x | 4 concurrent ops |
| **Pinned Memory Speedup** | 2-3x | vs regular transfer |
| **Memory Growth** | < 10 MB | 1000 iterations |
| **Throughput** | > 100 buffers/sec | GPU mode |

## Continuous Integration

### GitHub Actions Workflow
```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.12'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-asyncio pytest-cov
      - name: Run tests
        run: ./run_tests.sh cpu  # CI doesn't have GPU
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

## Troubleshooting

### Tests Fail on CUDA
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Run CPU-only tests
CUDA_VISIBLE_DEVICES="" python -m pytest tests/ -v
```

### Import Errors
```bash
# Add project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python -m pytest tests/ -v
```

### Slow Tests
```bash
# Skip slow memory leak tests
python -m pytest tests/ -v -m "not slow"
```

### Debug Specific Test
```bash
# Run with full output and stop on first failure
python -m pytest tests/test_cuda_streams.py -v -s -x
```

## Test Development Guidelines

### Adding New Tests

1. **Choose appropriate file**:
   - CUDA/GPU features → `test_cuda_streams.py`
   - Error handling → `test_error_recovery.py`
   - Memory issues → `test_memory_leaks.py`
   - End-to-end → `test_integration_comprehensive.py`

2. **Use fixtures**:
```python
@pytest.fixture
def radar_app():
    app = RadarApp(simulate=True, enable_viz=False)
    yield app
    if app.running:
        asyncio.run(app.stop_async())
```

3. **Mark CUDA tests**:
```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
async def test_gpu_feature():
    ...
```

4. **Mark slow tests**:
```python
@pytest.mark.slow
async def test_long_running():
    ...
```

5. **Use async for async code**:
```python
@pytest.mark.asyncio
async def test_async_method():
    await app.process_buffer(...)
```

## Code Coverage

Generate coverage report:
```bash
./run_tests.sh coverage

# View HTML report
open htmlcov/index.html
```

Target: **> 85% coverage** on core modules.

## Related Documentation

- [AGENTS.md](../AGENTS.md) - Development guidelines
- [README.md](../README.md) - Project overview
- [config.py](../config.py) - Configuration reference

## Contact

For test failures or questions, check:
1. System compatibility (GPU drivers, PyTorch version)
2. Configuration settings in `config.py`
3. Recent changes to core modules
4. GitHub Issues for known problems
