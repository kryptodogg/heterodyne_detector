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
    'wavelength': None,  # Computed from center_freq at runtime
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
    'memory_fraction': 0.8,     # Use 80% of GPU memory
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
    
    # Dataset structure
    'datasets': {
        'patterns': {
            'shape': (None, 13),  # (N, n_mfcc)
            'dtype': 'float32',
            'chunks': (100, 13),
            'compression': 'gzip',
            'compression_opts': 6,
        },
        'metadata': {
            'attrs': [
                'timestamp',
                'freq_offset',
                'detection_score',
                'doa_estimate',
                'snr_db'
            ]
        }
    },
    
    # Storage limits
    'max_patterns_per_session': 10000,
    'max_session_size_gb': 2.0,
    'auto_cleanup_days': 30,
    
    # Access
    'cache_size_mb': 100,
    'read_buffer_mb': 10,
}


# ============================================================
# VISUALIZATION - Dash Dashboard Settings
# ============================================================

VISUALIZATION = {
    # Dashboard
    'refresh_rate_hz': 60,        # Target 60 Hz (16.67 ms)
    'actual_refresh_ms': 50,      # Practical refresh (20 Hz)
    
    # Layout
    'grid': (2, 2),               # 2x2 grid
    'theme': 'plotly_dark',       # Dark theme
    
    # Plots
    'plots': {
        'rx1_spectrum': {
            'type': 'scatter',
            'title': 'RX1 Spectrum',
            'xaxis': 'Frequency (kHz)',
            'yaxis': 'Power (dB)',
        },
        'rx2_spectrum': {
            'type': 'scatter',
            'title': 'RX2 Spectrum',
            'xaxis': 'Frequency (kHz)',
            'yaxis': 'Power (dB)',
        },
        'mfcc_heatmap': {
            'type': 'heatmap',
            'title': 'MFCC Radar Features',
            'xaxis': 'Time (frames)',
            'yaxis': 'MFCC Coefficient',
            'colorscale': 'Viridis',
        },
        'doa_polar': {
            'type': 'scatterpolar',
            'title': 'Direction of Arrival',
            'angular': 'Angle (deg)',
            'radial': 'Power (dB)',
        },
    },
    
    # Performance
    'max_history_points': 1000,   # Rolling buffer
    'decimation_factor': 10,      # Decimate for display
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
# LOGGING
# ============================================================

LOGGING = {
    'level': 'INFO',              # DEBUG, INFO, WARNING, ERROR
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': './radar_app.log',
    'max_size_mb': 100,
    'backup_count': 5,
}


# ============================================================
# TESTING
# ============================================================

TESTING = {
    'simulation': {
        'num_targets': 3,
        'target_velocities': [5.0, -3.0, 10.0],  # m/s
        'target_angles': [0.0, 45.0, -30.0],     # degrees
        'snr_db': 20.0,
        'heterodyne_enabled': True,
        'heterodyne_freq': 5e3,  # 5 kHz beat
    },
    
    'validation': {
        'max_processing_time_ms': 50.0,  # Must process in < 50ms
        'min_detection_rate': 0.8,        # 80% detection rate
        'max_false_positive_rate': 0.1,   # 10% false positives
    }
}


# ============================================================
# RUNTIME COMPUTED VALUES
# ============================================================

def update_wavelength(center_freq):
    """Update wavelength based on center frequency"""
    c = 3e8  # Speed of light
    RADAR_GEOMETRY['wavelength'] = c / center_freq
    return RADAR_GEOMETRY['wavelength']


def compute_baseline():
    """Compute baseline between RX elements"""
    rx1_pos = RADAR_GEOMETRY['RX1']['position']
    rx2_pos = RADAR_GEOMETRY['RX2']['position']
    return np.linalg.norm(rx2_pos - rx1_pos)


def get_doa_resolution():
    """Compute theoretical DOA resolution"""
    baseline = compute_baseline()
    wavelength = RADAR_GEOMETRY['wavelength']
    if wavelength is None:
        return None
    # Rayleigh criterion
    return np.degrees(wavelength / baseline)


def get_torch_geometry(device=None):
    """
    Convert RADAR_GEOMETRY to Torch tensors on specified device.
    
    Args:
        device: torch.device (default: CPU)
        
    Returns:
        dict: Geometry with positions as torch.Tensor
    """
    if device is None:
        device = torch.device('cpu')
        
    torch_geo = {}
    
    for key in ['TX1', 'TX2', 'RX1', 'RX2']:
        torch_geo[key] = {
            'position': torch.tensor(
                RADAR_GEOMETRY[key]['position'], 
                dtype=torch.float32, 
                device=device
            ),
            'orientation': RADAR_GEOMETRY[key]['orientation'].copy(),
            'gain_db': RADAR_GEOMETRY[key].get('gain_db', 0.0),
            'power_dbm': RADAR_GEOMETRY[key].get('power_dbm', 0.0)
        }
        
    # Add array properties
    torch_geo['array_type'] = RADAR_GEOMETRY['array_type']
    torch_geo['wavelength'] = RADAR_GEOMETRY['wavelength']
    
    return torch_geo


# ============================================================
# CONFIG VALIDATION
# ============================================================

def validate_config():
    """Validate configuration consistency"""
    errors = []
    
    # Check window/hop relationship
    if MFCC_RADAR_SETTINGS['hop_length'] > MFCC_RADAR_SETTINGS['window_size']:
        errors.append("hop_length must be <= window_size")
    
    # Check Doppler range vs sample rate
    max_doppler = MFCC_RADAR_SETTINGS['fmax']
    nyquist = GPU_CONFIG['sample_rate'] / 2
    if max_doppler > nyquist:
        errors.append(f"fmax ({max_doppler}) exceeds Nyquist ({nyquist})")
    
    # Check GPU memory
    required_gb = GPU_CONFIG['memory_pool_gb']
    if required_gb < 2.0:
        errors.append("memory_pool_gb should be at least 2.0 GB")
    
    if errors:
        print("❌ Configuration errors:")
        for err in errors:
            print(f"  - {err}")
        return False
    
    return True


# Initialize
if __name__ == "__main__":
    print("Configuration Validation")
    print("="*60)
    
    # Update wavelength for default frequency
    wl = update_wavelength(GPU_CONFIG['center_freq'])
    print(f"Wavelength: {wl*1000:.2f} mm")
    
    # Baseline
    baseline = compute_baseline()
    print(f"Baseline: {baseline*100:.2f} cm")
    
    # DOA resolution
    doa_res = get_doa_resolution()
    if doa_res:
        print(f"DOA Resolution: {doa_res:.2f}°")
    
    # Validate
    print()
    if validate_config():
        print("✅ Configuration valid")
    else:
        print("❌ Configuration has errors")
