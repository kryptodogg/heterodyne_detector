# Implementation Plan: Refactor Core Pipeline for Torch-First Performance

## Phase 1: Preparation & Scaffolding
- [ ] Task: Initialize testing environment for core signal processing
    - [ ] Create `tests/test_signal_math.py` to compare NumPy vs Torch results
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Preparation & Scaffolding' (Protocol in workflow.md)

## Phase 2: Configuration & Geometry Refactor
- [ ] Task: Update `config.py` for Torch compatibility
    - [ ] Add Torch-based pre-calculation helpers
- [ ] Task: Implement `RadarGeometry` in `main.py` or a local module
    - [ ] Create `RadarGeometry` dataclass with Torch tensor positions
    - [ ] Verify steering vector math in Torch
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Configuration & Geometry Refactor' (Protocol in workflow.md)

## Phase 3: Main Pipeline Refactor
- [ ] Task: Refactor `RadarApp.__init__` for optimized GPU setup
    - [ ] Pre-allocate Torch buffers for signal ingress
- [ ] Task: Update `RadarApp.process_buffer` to use Torch-first logic
    - [ ] Remove redundant NumPy conversions
    - [ ] Ensure all sub-modules (detector, canceller) receive Torch tensors
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Main Pipeline Refactor' (Protocol in workflow.md)

## Phase 4: Performance & Integration
- [ ] Task: Benchmark 10 MSPS throughput
    - [ ] Measure end-to-end latency on GPU
- [ ] Task: Final system integration test with simulation mode
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Performance & Integration' (Protocol in workflow.md)
