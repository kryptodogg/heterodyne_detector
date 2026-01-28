# Specification: Refactor Core Pipeline for Torch-First Performance

## Overview
This track focuses on transforming the current radar pipeline into a Torch-first architecture. The goal is to move all heavy numerical computations from NumPy to PyTorch, specifically targeting the AMD GPU (ROCm) for 10 MSPS real-time throughput.

## Scope
- **Geometry Integration:** Implement the `RadarGeometry` class directly within the pipeline (replacing the missing `geometry.py`).
- **Data Flow:** Refactor `main.py` to handle `torch.Tensor` objects natively, minimizing CPU-GPU copies.
- **Config Optimization:** Update `config.py` to support Torch-centric pre-calculations.
- **Performance:** Ensure the processing loop can sustain 10 MSPS without dropping buffers.

## Requirements
- Use `torch.from_numpy()` for initial SDR data ingress.
- Implement beamforming and steering vector logic using Torch operations.
- Maintain existing MFCC and Pattern Matching logic but ensure they receive Torch tensors.
- Modules must remain under 500 lines.

## Success Criteria
- Pipeline processes 10 MHz buffers in <50ms.
- 60 Hz visualization remains fluid.
- Automated tests verify correctness of Torch-based signal math vs NumPy legacy.
