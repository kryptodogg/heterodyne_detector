"""
config.py - Centralized Configuration
Radar geometry, MFCC radar settings, GPU memory plan, HDF5 organization
"""

import numpy as np
import torch

# ============================================================
# RADAR GEOMETRY - 2TX2RX Configuration
# ============================================================

RADAR_GEOMETRY = {
    # TX1 - Transmitter 1
    'TX1': {
        'position': np.array([0.0, 0.0, 0.0]),  # meters (x, y, z)
        'orientation': {
            'yaw': 0.0,    # degrees
            'pitch': 0.0,  # degrees
            'roll': 0.0    # degrees
        },
        'power_dbm': 0.0,  # TX power
    },
    
    # TX2 - Transmitter 2
    'TX2': {
        'position': np.array([0.15, 0.0, 0.0]),  # 15 cm spacing
        'orientation': {
            'yaw': 0.0,
            'pitch': 0.0,
            'roll': 0.0
        },
        'power_dbm': 0.0,
    },
    
    # RX1 - Receiver 1
    'RX1': {
        'position': np.array([0.0, 0.10, 0.0]),  # 10 cm offset in Y
        'orientation': {
            'yaw': 0.0,
            'pitch': 0.0,
            'roll': 0.0
        },
        'gain_db': 50.0,
    },
    
    # RX2 - Receiver 2
    'RX2': {
        'position': np.array([0.15, 0.10, 0.0]),  # Diagonal from RX1
        'orientation': {
            'yaw': 0.0,
            'pitch': 0.0,
            'roll': 0.0
        },
        'gain_db': 50.0,
    },
    
    # Array properties
    'wavelength': 0.125,  # 2.4 GHz center freq approx
    'array_type': '2x2_planar',  # Array topology
}


# ============================================================
# NOISE CANCELLATION - Beamforming + Adaptive LMS
# ============================================================

NOISE_CANCELLATION = {
    # LMS adaptive filter parameters
    'filter_length': 64,        # Tap length
    'learning_rate': 0.01,      # Step size (mu)
    'convergence_threshold': 1e-5,
    
    # Beamforming parameters
    'num_steering_angles': 37,  # -90° to +90° in 5° steps
    'steering_resolution': 5.0, # degrees
    
    # Spatial filtering
    'use_beamforming': True,
    'use_adaptive_lms': True,
    'combine_method': 'weighted',  # 'weighted' or 'cascade'
    
    # Initialization
    'initial_weights': 'geometry',  # 'geometry' or 'identity' or 'zeros'
    'adapt_online': True,
    
    # Performance
    'batch_size': 1,  # Process N buffers at once
    'gpu_accelerated': True,
}


# ============================================================
# RANGE-DOPPLER - Velocity/Distance Processing
# ============================================================

RANGE_DOPPLER = {
    'range_fft_size': 512,      # 512 bins
    'doppler_fft_size': 128,    # 128 bins
    'num_cpis': 128,            # Coherent Processing Intervals
    'use_windowing': True,      # Hann windowing
    
    # CFAR Parameters
    'cfar_guard_cells': 4,      # Guard around test cell
    'cfar_training_cells': 16,  # Noise estimation region
    'cfar_pfa': 1e-4,           # Probability of false alarm
    
    # Scaling
    'max_range_m': 100.0,       # Max range for display
    'max_doppler_hz': 500.0,    # Max Doppler for display
}


# ============================================================
# PPI CONFIGURATION - Polar Display
# ============================================================

PPI_CONFIG = {
    'num_angles': 37,           # Same as beamforming
    'num_range_bins': 256,      # Range resolution
    'max_range_m': 100.0,       # Display limit
    'beamformer': 'conventional', # Conventional or MVDR
}


# ============================================================
# TARGET TRACKING - Kalman Persistence
# ============================================================

TRACKING = {
    'max_tracks': 20,           # Limit active tracks
    'gate_threshold': 5.0,      # Mahalanobis gating
    'track_timeout': 10,        # Frames to keep lost track
    'min_hits_for_track': 2,    # Confirmation threshold
    'dt': 0.05,                 # 20 Hz update rate
    
    # Noise parameters
    'process_noise': 0.1,
    'measurement_noise': 1.0,
    'init_velocity_uncertainty': 10.0,
}


# ============================================================
# MFCC RADAR SETTINGS - Doppler-Optimized Features
# ============================================================

MFCC_RADAR_SETTINGS = {
    # Time-frequency analysis
    'window_size': 0.375,       # seconds (375 ms)
    'hop_length': 0.05,         # seconds (50 ms, 20 Hz frame rate)
    'window_type': 'hann',      # Window function
    
    # FFT parameters
    'n_fft': 2048,              # FFT size
    'n_mfcc': 13,               # Number of MFCC coefficients
    
    # Mel filterbank
    'n_mels': 40,               # Number of mel bands
    'fmin': 0.0,                # Hz (include DC for radar)
    'fmax': 500.0,              # Hz (Doppler range)
    
    # Radar-specific
    'doppler_range': (10, 500), # Hz (human motion range)
    'dc_suppress': False,       # Keep DC for stationary targets
    'normalize': True,          # Normalize features
    
    # Processing
    'use_delta': True,          # Include delta (velocity) features
    'use_delta_delta': False,   # Include acceleration features
    'lifter': 22,               # Cepstral liftering
}


# ============================================================
# GPU CONFIGURATION - Memory and Compute
# ============================================================

GPU_CONFIG = {
    # Device settings
    'device': 'cuda:0',         # CUDA device
    # 'memory_fraction': 0.8,     # Use 80% of GPU memory
    'memory_pool_gb': 4.0,      # Pre-allocate 4 GB pool
    
    # Compute settings
    'use_mixed_precision': True,  # FP16 where possible
    'cudnn_benchmark': True,      # Optimize for fixed input sizes
    'num_workers': 4,             # Data loading threads
    
    # Buffer settings
    'buffer_size': 2**16,         # 65536 samples
    'batch_size': 1,              # Buffers per processing batch
    'prefetch_buffers': 2,        # Async buffer prefetch
    
    # SDR settings
    'sample_rate': 10e6,          # 10 MHz (10 MSPS)
    'center_freq': 2.4e9,         # 2.4 GHz default
    
    # Performance
    'zero_copy': True,            # Zero-copy GPU transfers
    'async_processing': True,     # Async GPU operations
}


# ============================================================
# HDF5 STORAGE - Pattern Library Organization
# ============================================================

HDF5_STORAGE = {
    # Paths
    'base_path': './radar_libraries',
    'temp_path': './radar_libraries/temp',
    
    # Organization hierarchy
    'hierarchy': [
        'date',       # YYYY-MM-DD
        'session',    # Session_HHMMSS
        'band'        # e.g., 2400MHz, 915MHz
    ],
}

LIBRARY_CONFIG = HDF5_STORAGE

DATASET_CONFIG = {
    'patterns': {
        'shape': (None, 13),  # (N, n_mfcc)
        'dtype': 'float32',
        'chunks': (100, 13),
        'compression': 'gzip',
        'compression_opts': 6,
    },
    'metadata': {
        'columns': ['timestamp', 'score', 'doa', 'freq_offset'],
        'dtype': 'float64'
    }
}

PREDEFINED_LABELS = ['human', 'vehicle', 'noise', 'interference']

ISM_BANDS = {
    '915MHz': {'center_freq': 915e6, 'bandwidth': 26e6},
    '2400MHz': {'center_freq': 2.4e9, 'bandwidth': 80e6},
    '5800MHz': {'center_freq': 5.8e9, 'bandwidth': 150e6}
}


# ============================================================
# VISUALIZATION - Dash Dashboard Settings
# ============================================================

VISUALIZATION = {
    # Dashboard
    'refresh_rate_hz': 60,        # Target 60 Hz (16.67 ms)
    'actual_refresh_ms': 50,      # Practical refresh (20 Hz)
    
    # Layout
    'grid': (3, 3),               # 3x3 grid
    'theme': 'plotly_dark',       # Dark theme
}


# ============================================================
# DETECTION THRESHOLDS
# ============================================================

DETECTION = {
    'heterodyne_threshold': 0.7,
    'voice_threshold': 0.15,
    'pattern_match_threshold': 0.85,
    'snr_threshold_db': 10.0,
}


# ============================================================
# RUNTIME COMPUTED VALUES
# ============================================================

def get_torch_geometry(device=None):
    """
    Convert RADAR_GEOMETRY to Torch tensors on specified device.
    """
    if device is None:
        device = torch.device('cpu')
        
    torch_geo = {}
    for key in ['TX1', 'TX2', 'RX1', 'RX2']:
        torch_geo[key] = torch.tensor(
            RADAR_GEOMETRY[key]['position'], 
            dtype=torch.float32, 
            device=device
        )
    return torch_geo