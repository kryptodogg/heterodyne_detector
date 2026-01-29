# GEMINI.md - Radar Application Context

## Project Overview
This project is a high-performance **2TX2RX Radar Application** built with a **Torch-first architecture**. It is designed for real-time signal processing at 10 MSPS, utilizing **AMD GPU acceleration (ROCm)** via PyTorch.

The core objective is to detect heterodyne artifacts and "phantom voices" using physical antenna geometry to drive spatial filtering (beamforming) and adaptive noise cancellation.

### Core Technologies
- **Language:** Python 3.12+ (managed via Conda)
- **Signal Processing:** PyTorch (with ROCm 7.1 support), torchaudio, SciPy
- **Hardware Interface:** Pluto+ SDR via `pyadi-iio` and `libiio`
- **Visualization:** Dash by Plotly (Real-time 3x3 dashboard at 20-60 Hz)
- **Data Storage:** HDF5 for hierarchical pattern libraries
- **Framework:** Conductor (Spec-driven development protocol)

### System Architecture
The application follows a modular pipeline orchestrated by `main.py`:
1.  **SDR Interface (`sdr_interface.py`):** Ingress of dual-channel IQ samples (hardware or simulation).
2.  **Noise Cancellation (`noise_canceller.py`):** Spatial beamforming (MVDR) and temporal adaptive filtering (LMS).
3.  **Heterodyne Detection (`heterodyne_detector.py`):** Multi-stage radio-to-acoustic downconversion and phantom voice analysis.
4.  **Range-Doppler Processing (`range_doppler.py`):** Velocity and distance estimation via FFTs and CFAR detection.
5.  **PPI Display (`ppi_processor.py`):** Polar Position Indicator for angular scanning.
6.  **Target Tracking (`tracker.py`):** Kalman filter-based multi-target persistence.
7.  **Feature Extraction (`audio_processor.py`):** torchaudio-based MFCCs and voice quality metrics.
8.  **Pattern Matching (`pattern_matcher.py`):** GPU-accelerated signal classification.
9.  **Visualization (`visualizer.py`):** Real-time Dash dashboard.

## Building and Running

### Environment Setup
The project uses a **Conda base environment** with a specific PyTorch build for AMD GPUs:
```bash
# Verify Torch and ROCm
conda run -n base python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Expected: 2.10.0+rocm7.1 True
```

### Key Commands
- **Run Simulation:** `conda run -n base python main.py --simulate --duration 10`
- **Run with Hardware:** `conda run -n base python main.py --freq 2400 --sample-rate 10`
- **Run Tests:** `PYTHONPATH=. conda run -n base python -m unittest discover tests/`
- **Diagnostics:** `python diagnose.py` or `rocm-smi`

## Development Conventions

### Torch-First Principles
- **Zero-Copy Pipeline:** Prioritize `torch.Tensor` over NumPy. Keep data on GPU memory as long as possible.
- **Device Consistency:** Always verify tensors are on the same device (e.g., `cuda:0`) before operations.
- **Pre-allocation:** Pre-allocate buffers in `__init__` to avoid fragmentation during real-time loops.

### Coding Style
- **Modularity:** Modules should remain under 500 lines when possible.
- **Geometry-Driven:** All spatial algorithms must use the physical positions defined in `config.py:RADAR_GEOMETRY`.
- **Documentation:** Follow the **Conductor Workflow**. Specifications reside in `conductor/tracks/` and work is tracked in `plan.md`.

### Testing Practices
- **TDD:** Follow the "Red-Green-Refactor" cycle.
- **Mocking:** Use simulation modes or mock objects for hardware-dependent tests.
- **Environment:** If hardware/GPU is unstable, run tests on CPU using `CUDA_VISIBLE_DEVICES=""`.

## Key Files
- `main.py`: Entry point and pipeline orchestrator.
- `config.py`: Centralized configuration for geometry, algorithms, and GPU limits.
- `conductor/`: Project governance, specifications, and implementation plans.
- `AGENTS.md`: Detailed development guidelines and architectural principles.
- `tests/`: Comprehensive unit test suite for signal math and module logic.

## AI Agent Toolkit (MCP)
This project leverages several Model Context Protocol (MCP) servers and expert skills to enhance AI-assisted development:

- **Serena (`serena`):** Professional coding agent providing semantic tools for resource-efficient codebase analysis, symbolic edits, and task adherence. Use Serena's symbolic search over raw file reads for token efficiency.
- **Open Aware (`open-aware`):** Provides code intelligence and context retrieval over pre-indexed open-source repositories. Useful for comparing implementations or finding patterns in libraries like `PyTorch` or `librosa`.
- **Skillshare (`skillshare`):** Manages and synchronizes AI skills across different platforms (Claude, Cursor, Gemini). Use `skillshare sync` to distribute expert radar logic across your tools.
- **Radar Expert (`radar-expert`):** A specialized skill for advanced signal processing. Call `activate_skill name="radar-expert"` (or refer to its managed path) whenever complex radar math, beamforming, or heterodyning logic is required. **Crucial:** Proactively expand this skill with new patterns, physics insights, or GPU optimizations learned during development to maximize future time savings.
