# Implementation Plan: Refactor Core Pipeline for Torch-First Performance

## Phase 1: Preparation & Scaffolding [checkpoint: 3f74054]
- [x] Task: Initialize testing environment for core signal processing [commit: 5a2a17d]
    - [ ] Create `tests/test_signal_math.py` to compare NumPy vs Torch results
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Preparation & Scaffolding' (Protocol in workflow.md)

## Phase 2: Configuration & Geometry Refactor [checkpoint: 5de0980]
- [x] Task: Update `config.py` for Torch compatibility [commit: 106e695]
    - [ ] Add Torch-based pre-calculation helpers
- [x] Task: Implement `RadarGeometry` in `main.py` or a local module [commit: 715b586]
    - [ ] Create `RadarGeometry` dataclass with Torch tensor positions
    - [ ] Verify steering vector math in Torch
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Configuration & Geometry Refactor' (Protocol in workflow.md)

## Phase 3: Main Pipeline Refactor
- [x] Task: Refactor `RadarApp.__init__` for optimized GPU setup [commit: b6baee2]
    - [ ] Pre-allocate Torch buffers for signal ingress
- [x] Task: Update `RadarApp.process_buffer` to use Torch-first logic [commit: 3ee2954]
    - [ ] Remove redundant NumPy conversions
    - [ ] Ensure all sub-modules (detector, canceller) receive Torch tensors
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Main Pipeline Refactor' (Protocol in workflow.md)

## Phase 4: Performance & Integration
- [ ] Task: Benchmark 10 MSPS throughput
    - [ ] Measure end-to-end latency on GPU
- [ ] Task: Final system integration test with simulation mode
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Performance & Integration' (Protocol in workflow.md)
