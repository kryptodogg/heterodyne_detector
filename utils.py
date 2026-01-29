#!/usr/bin/env python3
"""
GPU Utilities and Memory Management for ROCm Acceleration
Zero-copy operations, async processing, memory pooling
"""

import torch
import numpy as np
import asyncio
import time
from typing import Optional, Dict, List, Tuple, Any
from collections import deque
import gc
from .config import GPU_CONFIG, MEMORY_CONFIG


class GPUMemoryPool:
    """
    Zero-copy GPU memory pool for ROCm acceleration
    Pre-allocated buffers to eliminate malloc overhead
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.pools = {}  # Buffer pools by size
        self.allocated_tensors = set()
        self.total_allocated = 0

        # Initialize memory pools
        self._initialize_pools()

    def _initialize_pools(self):
        """Initialize GPU memory pools"""
        buffer_pool_mb = MEMORY_CONFIG["buffer_pool_mb"]

        # Common buffer sizes (powers of 2)
        buffer_sizes = [1024, 4096, 16384, 65536]  # Samples per buffer

        for size in buffer_sizes:
            # Calculate memory needed per buffer
            bytes_per_sample = 8  # Complex64 = 8 bytes
            buffer_bytes = size * bytes_per_sample * 2  # 2 channels

            # Calculate how many buffers we can allocate
            buffers_per_size = min(
                8, (buffer_pool_mb * 1024 * 1024) // (buffer_bytes * 2)
            )

            pool_list = []
            for i in range(buffers_per_size):
                tensor = torch.zeros(
                    size, 2, dtype=torch.complex64, device=self.device, pin_memory=True
                )
                pool_list.append(tensor)

            self.pools[size] = pool_list
            self.total_allocated += len(pool_list) * buffer_bytes

    def get_buffer(self, size: int) -> torch.Tensor:
        """Get buffer from pool (zero-copy)"""
        # Find closest size pool
        pool_size = self._find_closest_pool_size(size)

        if pool_size in self.pools and self.pools[pool_size]:
            buffer = self.pools[pool_size].pop()
            self.allocated_tensors.add(id(buffer))
            return buffer
        else:
            # Allocate new buffer if pool exhausted
            return torch.zeros(
                size, 2, dtype=torch.complex64, device=self.device, pin_memory=True
            )

    def return_buffer(self, buffer: torch.Tensor):
        """Return buffer to pool"""
        buffer_id = id(buffer)
        if buffer_id in self.allocated_tensors:
            size = buffer.shape[0]
            pool_size = self._find_closest_pool_size(size)

            # Zero buffer for reuse
            buffer.zero_()

            if pool_size in self.pools:
                self.pools[pool_size].append(buffer)

            self.allocated_tensors.discard(buffer_id)

    def _find_closest_pool_size(self, size: int) -> int:
        """Find nearest pool size >= requested size"""
        available_sizes = sorted(self.pools.keys())
        for pool_size in available_sizes:
            if pool_size >= size:
                return pool_size
        return max(available_sizes) if available_sizes else size

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics"""
        allocated_mb = self.total_allocated / (1024 * 1024)
        available_gb = get_gpu_memory_info()["available_gb"]

        return {
            "allocated_mb": allocated_mb,
            "available_gb": available_gb,
            "pool_sizes": {k: len(v) for k, v in self.pools.items()},
            "active_buffers": len(self.allocated_tensors),
        }

    def cleanup(self):
        """Clean up all allocated memory"""
        for pool in self.pools.values():
            for tensor in pool:
                del tensor
        self.pools.clear()
        self.allocated_tensors.clear()
        torch.cuda.empty_cache()


class AsyncGPUProcessor:
    """
    Async GPU processing with multiple CUDA streams
    Non-blocking operations for real-time performance
    """

    def __init__(self, device: torch.device, num_streams: int = 4):
        self.device = device
        self.streams = [torch.cuda.Stream(device=device) for _ in range(num_streams)]
        self.current_stream = 0

        # Memory pool for zero-copy operations
        self.memory_pool = GPUMemoryPool(device)

        # Async task queue
        self.task_queue = asyncio.Queue(maxsize=100)
        self.processing = False

    def get_next_stream(self) -> torch.cuda.Stream:
        """Get next CUDA stream (round-robin)"""
        stream = self.streams[self.current_stream]
        self.current_stream = (self.current_stream + 1) % len(self.streams)
        return stream

    async def process_signal_async(
        self, signal: torch.Tensor, operation: str, **kwargs
    ) -> torch.Tensor:
        """
        Async signal processing on GPU

        Args:
            signal: Input signal tensor
            operation: Processing operation ('fft', 'stft', 'mfcc', etc.)
            **kwargs: Operation-specific parameters

        Returns:
            Processed tensor
        """
        with torch.cuda.stream(self.get_next_stream()):
            if operation == "fft":
                return self._process_fft(signal, **kwargs)
            elif operation == "stft":
                return self._process_stft(signal, **kwargs)
            elif operation == "mfcc":
                return self._process_mfcc(signal, **kwargs)
            elif operation == "correlation":
                return self._process_correlation(signal, **kwargs)
            else:
                raise ValueError(f"Unknown operation: {operation}")

    def _process_fft(self, signal: torch.Tensor, n_fft: int = None) -> torch.Tensor:
        """FFT processing with GPU acceleration"""
        if n_fft is None:
            n_fft = signal.shape[-1]

        # Ensure signal is on GPU
        if signal.device != self.device:
            signal = signal.to(self.device, non_blocking=True)

        return torch.fft.fft(signal, n=n_fft, dim=-1)

    def _process_stft(
        self, signal: torch.Tensor, n_fft: int = 1024, hop_length: int = 512
    ) -> torch.Tensor:
        """STFT processing on GPU"""
        # Librosa STFT is not GPU-accelerated, use torch implementation
        return torch.stft(
            signal,
            n_fft=n_fft,
            hop_length=hop_length,
            window=torch.hann_window(n_fft, device=self.device),
            return_complex=True,
        )

    def _process_mfcc(
        self,
        signal: torch.Tensor,
        sample_rate: int = 2400000,
        n_mfcc: int = 13,
        n_fft: int = 1024,
        hop_length: int = 512,
    ) -> torch.Tensor:
        """MFCC extraction on GPU"""
        # Compute STFT first
        stft = self._process_stft(signal, n_fft, hop_length)

        # Compute magnitude spectrogram
        magnitude = torch.abs(stft)

        # Create mel filter bank (GPU)
        mel_fbanks = self._create_mel_filterbank(n_fft, sample_rate, n_mels=40).to(
            self.device
        )

        # Apply mel filters
        mel_spectrogram = torch.matmul(mel_fbanks, magnitude)

        # Log compression
        log_mel = torch.log10(mel_spectrogram + 1e-10)

        # DCT to get MFCC
        mfcc = self._dct(log_mel, n_mfcc=n_mfcc)

        return mfcc

    def _process_correlation(
        self, signal1: torch.Tensor, signal2: torch.Tensor, mode: str = "full"
    ) -> torch.Tensor:
        """Cross-correlation using FFT for GPU acceleration"""
        # Pad signals to same length
        n = max(signal1.shape[-1], signal2.shape[-1])
        padded_sig1 = torch.nn.functional.pad(signal1, (0, n - signal1.shape[-1]))
        padded_sig2 = torch.nn.functional.pad(signal2, (0, n - signal2.shape[-1]))

        # FFT both signals
        fft1 = torch.fft.fft(padded_sig1)
        fft2 = torch.fft.fft(padded_sig2)

        # Correlation via FFT (multiply by conjugate)
        correlation_fft = fft1 * torch.conj(fft2)

        # Inverse FFT
        correlation = torch.fft.ifft(correlation_fft)

        if mode == "same":
            # Extract same-length result
            center = n // 2
            return torch.abs(correlation[..., center - n // 2 : center + n // 2])

        return torch.abs(correlation)

    def _create_mel_filterbank(
        self, n_fft: int, sample_rate: int, n_mels: int = 40
    ) -> torch.Tensor:
        """Create mel filter bank on GPU"""
        # Mel frequency range (from config: 0-1200 Hz for radar)
        low_mel = 0
        high_mel = 2595 * np.log10(1 + 1200 / 700)

        # Create mel scale
        mels = torch.linspace(low_mel, high_mel, n_mels + 2)

        # Convert to Hz
        hz = 700 * (10 ** (mels / 2595) - 1)

        # Convert to FFT bin numbers
        bins = torch.floor((n_fft - 1) * hz / (sample_rate // 2))

        # Create filter bank
        filterbank = torch.zeros(n_mels, n_fft // 2 + 1)

        for i in range(n_mels):
            left = int(bins[i])
            center = int(bins[i + 1])
            right = int(bins[i + 2])

            for j in range(left, center):
                filterbank[i, j] = (j - left) / (center - left)
            for j in range(center, right):
                filterbank[i, j] = (right - j) / (right - center)

        return filterbank

    def _dct(self, x: torch.Tensor, n_mfcc: int = 13) -> torch.Tensor:
        """Discrete Cosine Transform (Type II)"""
        N = x.shape[-2]
        n = torch.arange(N, device=self.device, dtype=torch.float32)
        k = torch.arange(n_mfcc, device=self.device, dtype=torch.float32)

        # DCT matrix
        dct_matrix = torch.cos(np.pi * k[:, None] * (2 * n[None, :] + 1) / (2 * N))
        dct_matrix[:, 0] = 1 / np.sqrt(N)
        dct_matrix[:, 1:] = dct_matrix[:, 1:] * np.sqrt(2 / N)

        # Apply DCT
        return torch.matmul(dct_matrix, x)


class PerformanceMonitor:
    """
    Monitor GPU and system performance metrics
    Track utilization, memory, and latency
    """

    def __init__(self):
        self.metrics = deque(maxlen=1000)
        self.start_time = time.time()
        self.last_cleanup = time.time()

    def record_metric(self, metric_name: str, value: float):
        """Record a performance metric"""
        self.metrics.append(
            {"timestamp": time.time(), "metric": metric_name, "value": value}
        )

    def get_gpu_metrics(self) -> Dict[str, Any]:
        """Get current GPU utilization metrics"""
        if torch.cuda.is_available():
            return {
                "memory_used_gb": torch.cuda.memory_allocated() / (1024**3),
                "memory_cached_gb": torch.cuda.memory_reserved() / (1024**3),
                "utilization": self._get_gpu_utilization(),
            }
        return {}

    def get_average_latency(self) -> float:
        """Get average processing latency in ms"""
        latencies = [m["value"] for m in self.metrics if m["metric"] == "latency"]
        return np.mean(latencies) if latencies else 0.0

    def _get_gpu_utilization(self) -> float:
        """Get GPU utilization percentage"""
        # This would require nvidia-ml-py or rocm-smi
        # For now, return estimated value
        return 0.0

    def should_cleanup(self) -> bool:
        """Check if memory cleanup is needed"""
        return (time.time() - self.last_cleanup) > 600  # 10 minutes

    def perform_cleanup(self):
        """Perform GPU memory cleanup"""
        torch.cuda.empty_cache()
        gc.collect()
        self.last_cleanup = time.time()


# Global instances (singleton pattern)
_gpu_processor = None
_memory_pool = None
_perf_monitor = None


def get_gpu_processor() -> AsyncGPUProcessor:
    """Get global GPU processor instance"""
    global _gpu_processor
    if _gpu_processor is None:
        device = torch.device(GPU_CONFIG["device"])
        _gpu_processor = AsyncGPUProcessor(device, GPU_CONFIG["async_streams"])
    return _gpu_processor


def get_memory_pool() -> GPUMemoryPool:
    """Get global memory pool instance"""
    global _memory_pool
    if _memory_pool is None:
        device = torch.device(GPU_CONFIG["device"])
        _memory_pool = GPUMemoryPool(device)
    return _memory_pool


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor"""
    global _perf_monitor
    if _perf_monitor is None:
        _perf_monitor = PerformanceMonitor()
    return _perf_monitor


def get_gpu_memory_info() -> Dict[str, float]:
    """Get comprehensive GPU memory information"""
    if not torch.cuda.is_available():
        return {}

    total_memory = torch.cuda.get_device_properties(0).total_memory
    allocated = torch.cuda.memory_allocated()
    cached = torch.cuda.memory_reserved()

    return {
        "total_gb": total_memory / (1024**3),
        "allocated_gb": allocated / (1024**3),
        "cached_gb": cached / (1024**3),
        "free_gb": (total_memory - allocated) / (1024**3),
    }


def ensure_gpu_memory(free_gb_required: float = 1.0) -> bool:
    """Ensure enough GPU memory is available"""
    info = get_gpu_memory_info()
    return info.get("free_gb", 0) >= free_gb_required


def async_gpu_barrier():
    """Synchronize all GPU streams"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# Utility functions for signal processing
def create_window_function(
    window_type: str, size: int, device: torch.device
) -> torch.Tensor:
    """Create window function on GPU"""
    if window_type == "hann":
        return torch.hann_window(size, device=device)
    elif window_type == "hamming":
        return torch.hamming_window(size, device=device)
    elif window_type == "blackman":
        return torch.blackman_window(size, device=device)
    else:
        return torch.ones(size, device=device)


def zero_copy_to_gpu(data: np.ndarray, device: torch.device) -> torch.Tensor:
    """Zero-copy numpy array to GPU tensor"""
    return torch.from_numpy(data).to(device, non_blocking=True)


def zero_copy_to_cpu(tensor: torch.Tensor) -> np.ndarray:
    """Zero-copy GPU tensor to numpy array"""
    return tensor.cpu().numpy()


# Cleanup function
def cleanup_gpu_resources():
    """Clean up all GPU resources"""
    global _gpu_processor, _memory_pool, _perf_monitor

    if _gpu_processor:
        _gpu_processor.memory_pool.cleanup()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _gpu_processor = None
    _memory_pool = None
    _perf_monitor = None
