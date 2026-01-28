# Technology Stack

## Core Technologies
- **Programming Language:** Python
- **AI/ML & Signal Processing:** PyTorch (ROCm for AMD GPU), Torchaudio, SciPy
- **SDR Interface:** Pluto SDR via `pyadi-iio`
- **Frontend/Dashboard:** Plotly Dash
- **Data Storage:** HDF5 (via `h5py`)
- **Infrastructure:** Git, Conda

## Key Principles
- **Torch-First:** Prioritize PyTorch tensors over NumPy arrays for all heavy computations to leverage GPU acceleration. Use `torch.from_numpy()` for necessary conversions.
