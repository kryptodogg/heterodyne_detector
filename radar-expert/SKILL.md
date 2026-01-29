---
name: radar-expert
description: GPU-accelerated radar/sonar signal processing with Torch-first architecture, heterodyning, and spatial noise cancellation via beamforming. Use when developing mmWave radar systems with ROCm-optimized processing pipelines, implementing adaptive signal cancellation, or working with real-time 3D signal visualization on AMD/NVIDIA hardware.
---

# Radar Expert

## Project Overview

Create a Claude skill for zero-copy GPU-accelerated radar/sonar signal processing using PyTorch 7.1 on AMD RX 6700 XT, specifically designed for heterodyning "phantom" voice signals and implementing heterodyning noise-cancellation algorithms. The system processes data from MR60BHA2 (60GHz) and HLK-LD2410_8592 (24GHz) mmWave modules with real-time 3D visualization.

**Core Technical Specifications:**

### Hardware Stack
- **GPU**: AMD RX 6700 XT (ROCm 5.6+ compatible)
- **Radar Modules**:
  - MR60BHA2 (60GHz, I2C/SPI/UART, bio-signal detection)
  - HLK-LD2410_8592 (24GHz, UART, presence/motion)
- **Communication**: Serial (USB-UART), Bluetooth (HC-05/ESP32 for wireless relay)

### Software Stack
- **PyTorch 7.1** with ROCm support
- **Zero-copy operations** via `torch.from_numpy()` with `pin_memory=True`
- **Plotly** for interactive 3D visualization
- **PySerial**, **PyBluez** for hardware communication
- **NumPy**, **SciPy** for signal preprocessing

**Key Capabilities:**
- Zero-copy GPU tensor pipelines (serial I/O → pinned memory → GPU)
- MVDR + adaptive LMS spatial noise cancellation
- IQ heterodyne detection with frequency offset estimation
- Frequency-domain beamforming and steering vector synthesis
- Real-time 3D visualization (Plotly + Dash)
- Multi-TX/RX antenna array processing (2TX2RX)
- Phantom voice extraction and cancellation
- Hardware interface for MR60BHA2 and HLK-LD2410_8592 modules

## Architecture Principles

### 1. Torch-First Data Flow

All data enters as PyTorch tensors on GPU. This eliminates NumPy-to-Torch conversion overhead:

```python
# ✅ Correct: Zero-copy pinned memory → GPU tensor
import torch
import numpy as np

# Allocate pinned memory
cpu_buffer = np.zeros((FRAME_SIZE,), dtype=np.complex64)
# Transfer with pin_memory optimization
gpu_tensor = torch.from_numpy(cpu_buffer).pin_memory().cuda(non_blocking=True)

# Processing stays on GPU (no CPU fallback)
result = torch.fft.fft(gpu_tensor)  # FFT stays on GPU
```

**Why This Matters:**
- `torch.from_numpy()` creates a view (no copy) if array is C-contiguous
- `pin_memory()` locks CPU memory → DMA transfers to GPU
- `non_blocking=True` allows CPU to continue while GPU transfer happens
- **Result**: 10-100x faster ingress vs. `torch.tensor()`

### 2. Zero-Copy GPU Processing Pipeline

Direct serial → pinned memory → GPU tensor transfer with CUDA streams for concurrent processing:

```python
class RadarProcessingSystem:
    def __init__(self):
        self.gpu_queue = torch.cuda.Stream()  # RX 6700 XT specific
        self.serial_manager = ConcurrentSerialManager()
        self.processing_graph = TorchJITGraph()  # Pre-compiled

    async def zero_copy_pipeline(self):
        # Serial → pinned memory → GPU (zero-copy)
        cpu_buffer = np.zeros((FRAME_SIZE,), dtype=np.complex64)
        gpu_tensor = torch.from_numpy(cpu_buffer).pin_memory().cuda(non_blocking=True)

        # Batch processing of radar frames with tensor operations
        processed_batch = self.process_radar_frames(gpu_tensor)

        # Memory-mapped files for large dataset handling
        return processed_batch
```

**Pipeline Features:**
- Direct serial → pinned memory → GPU tensor transfer
- CUDA streams for concurrent processing (RX 6700 XT optimization)
- Batch processing of radar frames with tensor operations
- Memory-mapped files for large dataset handling

### 3. Module Composition Pattern

RadarApp orchestrates specialized signal processors as composable modules:

```python
class RadarApp:
    def __init__(self):
        # Each module is independent, testable
        self.noise_canceller = SpatialNoiseCanceller(geometry, device)
        self.detector = HeterodyneDetector(sample_rate, device)
        self.processor = RangeDopplerProcessor(config, device)

    def process_buffer(self, rx1, rx2):
        # Linear pipeline: noise → detection → feature extraction
        clean_rx1, clean_rx2, info = self.noise_canceller.cancel(rx1, rx2)
        detection = self.detector.detect(clean_rx1, clean_rx2)
        features = self.audio_proc.extract_features(clean_rx1)
        return results
```

**Design Benefits:**
- Each module can be unit-tested independently
- Modules are reusable in different pipelines
- Easy to swap algorithms (e.g., RLS instead of LMS)
- Geometry dependencies are explicit

### 4. Geometry as First-Class Citizen

Radar geometry (antenna positions, baseline, wavelength) drives beamforming:

```python
@dataclass
class RadarGeometry:
    tx1_pos: torch.Tensor  # 3D position (x, y, z)
    rx1_pos: torch.Tensor
    rx2_pos: torch.Tensor
    baseline: float        # |rx2 - rx1|
    wavelength: float

    def compute_steering_vector(self, theta):
        """Generate steering vectors for angle theta"""
        k = 2 * np.pi / self.wavelength
        phase = k * self.baseline * torch.sin(theta)
        return torch.exp(1j * phase)
```

**Why:**
- Steering vectors depend on *exact* antenna layout
- Small geometry errors compound in beamformer null steering
- Tensor representation enables batch angle sweeps
- Geometric info passes to visualization (spatial heatmaps)

## Mathematical Core

### Continuous Wavelet Transform (CWT) with Morlet Wavelet
```python
def continuous_wavelet_transform(self, signal, scales, wavelet='morlet'):
    """
    Perform CWT for time-frequency analysis
    """
    # Generate Morlet wavelet
    wavelet_func = self.generate_morlet_wavelet(scales)

    # Convolve signal with scaled wavelets
    cwt_result = self.convolve_with_wavelets(signal, wavelet_func, scales)

    return cwt_result

def generate_morlet_wavelet(self, scales):
    """
    Generate Morlet wavelet function for given scales
    """
    # Morlet wavelet: g(t) = e^(-t²/2) * e^(i*ω₀*t)
    omega_0 = 5.0  # Dimensionless frequency parameter
    t = torch.linspace(-4, 4, 1024, device=self.device)
    morlet_real = torch.exp(-t**2 / 2) * torch.cos(omega_0 * t)
    morlet_imag = torch.exp(-t**2 / 2) * torch.sin(omega_0 * t)

    return torch.complex(morlet_real, morlet_imag)
```

### Hilbert-Huang Transform (HHT) for Non-Stationary Signal Decomposition
```python
def hilbert_huang_transform(self, signal):
    """
    Perform HHT for non-stationary signal decomposition
    """
    # Empirical Mode Decomposition (EMD)
    imfs = self.empirical_mode_decomposition(signal)

    # Apply Hilbert transform to each IMF
    hht_result = []
    for imf in imfs:
        analytic_imf = torch.fft.ifft(
            torch.fft.fft(imf) * torch.tensor([0, 2, 2, ..., 0])
        )
        instantaneous_amplitude = torch.abs(analytic_imf)
        instantaneous_phase = torch.angle(analytic_imf)
        instantaneous_frequency = torch.diff(instantaneous_phase) / (2 * torch.pi)

        hht_result.append({
            'amplitude': instantaneous_amplitude,
            'phase': instantaneous_phase,
            'frequency': instantaneous_frequency
        })

    return hht_result

def empirical_mode_decomposition(self, signal):
    """
    Perform EMD to decompose signal into Intrinsic Mode Functions (IMFs)
    """
    imfs = []
    residue = signal.clone()

    # Extract IMFs iteratively
    while not self.is_monotonic(residue) and len(imfs) < 10:
        imf = self.extract_imf(residue)
        imfs.append(imf)
        residue = residue - imf

    return imfs
```

### Maxwell's Equations Implementation for EM Propagation Modeling
```python
def solve_maxwells_equations(self, electric_field, magnetic_field, conductivity, permittivity):
    """
    Solve Maxwell's equations for EM propagation modeling
    """
    # Maxwell's equations in differential form:
    # ∇ × E = -∂B/∂t
    # ∇ × H = J + ∂D/∂t
    # ∇ · D = ρ
    # ∇ · B = 0

    # Using finite difference time domain (FDTD) method
    curl_e = self.curl_operator(electric_field)
    curl_h = self.curl_operator(magnetic_field)

    # Update fields using Yee algorithm
    updated_magnetic = magnetic_field - (self.dt / self.mu_0) * curl_e
    updated_electric = electric_field + (self.dt / (self.epsilon_0 * permittivity)) * (
        curl_h - conductivity * electric_field
    )

    return updated_electric, updated_magnetic

def curl_operator(self, field):
    """
    Compute curl of a 3D vector field
    """
    # Compute curl using finite differences
    dx_dy = torch.gradient(field[:, :, :], dim=1)[0]
    dy_dx = torch.gradient(field[:, :, :], dim=0)[0]

    # Return curl components
    return torch.stack([
        torch.gradient(field[:, :, 2], dim=1)[0] - torch.gradient(field[:, :, 1], dim=2)[0],
        torch.gradient(field[:, :, 0], dim=2)[0] - torch.gradient(field[:, :, 2], dim=0)[0],
        torch.gradient(field[:, :, 1], dim=0)[0] - torch.gradient(field[:, :, 0], dim=1)[0]
    ], dim=-1)
```

### Boundary Element Method for Cranial Acoustic Simulation
```python
def boundary_element_method(self, geometry, acoustic_params):
    """
    Apply boundary element method for cranial acoustic simulation
    """
    # Discretize boundary into elements
    boundary_elements = self.discretize_boundary(geometry)

    # Formulate integral equation system
    # ∫∫ G(x,y) ∂u(y)/∂n ds(y) = ∫∫ ∂G(x,y)/∂n u(y) ds(y) + f(x)
    # where G is Green's function, u is acoustic pressure

    # Assemble system matrix
    system_matrix = self.assemble_bem_matrix(boundary_elements, acoustic_params)

    # Solve for boundary values
    boundary_solution = torch.linalg.solve(system_matrix, acoustic_params.source_terms)

    return boundary_solution

def discretize_boundary(self, geometry):
    """
    Discretize the boundary into triangular elements
    """
    # Convert geometry to mesh
    vertices, triangles = self.geometry_to_mesh(geometry)

    # Calculate element properties
    element_areas = self.calculate_element_areas(triangles, vertices)
    element_normals = self.calculate_element_normals(triangles, vertices)

    return {
        'vertices': vertices,
        'triangles': triangles,
        'areas': element_areas,
        'normals': element_normals
    }
```

## Hardware Communication Layer

### MR60BHA2 Protocol Implementation
```python
class MR60BHA2Protocol:
    def __init__(self, connection_type='uart'):
        self.connection_type = connection_type
        self.fmcw_waveform_control = None

    def configure_fmcw_waveform(self, frequency, sweep_time, bandwidth):
        """
        Configure FMCW waveform parameters for MR60BHA2
        """
        # Set frequency, sweep time, and bandwidth
        cmd = self.build_fmcw_config_cmd(frequency, sweep_time, bandwidth)
        self.send_command(cmd)

    def build_fmcw_config_cmd(self, frequency, sweep_time, bandwidth):
        """
        Build FMCW configuration command
        """
        # Construct command bytes based on MR60BHA2 protocol
        cmd_bytes = bytearray()
        cmd_bytes.extend([0x01, 0x02])  # Command header
        cmd_bytes.extend(frequency.to_bytes(4, byteorder='little'))
        cmd_bytes.extend(sweep_time.to_bytes(4, byteorder='little'))
        cmd_bytes.extend(bandwidth.to_bytes(4, byteorder='little'))

        return cmd_bytes
```

### HLK-LD2410 Configuration
```python
class HLKLD2410Config:
    def __init__(self, baud_rate=256000):
        self.baud_rate = baud_rate
        self.presence_detection_params = {}

    def configure_presence_sensing(self, motion_threshold=0.1, static_threshold=0.05):
        """
        Configure presence sensing parameters for HLK-LD2410
        """
        # Set motion and static thresholds
        self.presence_detection_params['motion_threshold'] = motion_threshold
        self.presence_detection_params['static_threshold'] = static_threshold

        # Send configuration to device
        self.send_presence_config()

    def send_presence_config(self):
        """
        Send presence detection configuration to HLK-LD2410
        """
        # Construct configuration packet
        config_packet = self.build_presence_config_packet()

        # Send via UART
        self.uart_send(config_packet)
```

### Multi-Threaded Serial Data Acquisition with Circular Buffers
```python
import threading
import queue
from collections import deque

class ConcurrentSerialManager:
    def __init__(self, buffer_size=1024*1024):
        self.data_queue = queue.Queue(maxsize=100)
        self.circular_buffer = deque(maxlen=buffer_size)
        self.read_thread = None
        self.running = False

    def start_data_acquisition(self, port, baudrate):
        """
        Start multi-threaded serial data acquisition
        """
        self.serial_port = serial.Serial(port, baudrate, timeout=1)
        self.running = True
        self.read_thread = threading.Thread(target=self._read_loop)
        self.read_thread.start()

    def _read_loop(self):
        """
        Internal loop for reading data from serial port
        """
        while self.running:
            if self.serial_port.in_waiting > 0:
                data = self.serial_port.read(self.serial_port.in_waiting)
                self.circular_buffer.extend(data)

                # Put data in queue for processing
                try:
                    self.data_queue.put_nowait(data)
                except queue.Full:
                    pass  # Skip if queue is full

    def get_latest_data(self, num_bytes):
        """
        Get latest data from circular buffer
        """
        if len(self.circular_buffer) >= num_bytes:
            # Get the most recent num_bytes
            recent_data = bytes([self.circular_buffer[i] for i in range(-num_bytes, 0)])
            return recent_data
        else:
            return bytes(list(self.circular_buffer))
```

## Physics and Biophysics Knowledge Base

### Programming Skills
- Advanced PyTorch (CUDA ops, custom kernels, AMP)
- ROCm platform optimization for RDNA2 architecture
- Real-time systems programming (priority scheduling, buffer management)
- DSP implementation on GPU (FFT, filtering, convolution)
- Async I/O for hardware communication

### Mathematics & Signal Processing
- **Complex envelope detection** and IQ imbalance correction
- **Radar cross-section** calculations for biological tissues
- **Microwave acoustics**: EM-to-acoustic transduction physics
- **Nonlinear system identification** for neural interference patterns
- **Information theory** for covert signal embedding

### Physics & Biophysics

#### mmWave Propagation in Atmospheric and Biological Media
```python
def calculate_attenuation_coefficient(self, frequency, temperature, humidity):
    """
    Calculate atmospheric attenuation coefficient for mmWave signals
    """
    # Water vapor absorption
    water_vapor_attenuation = self.water_vapor_absorption(frequency, humidity)

    # Oxygen absorption
    oxygen_attenuation = self.oxygen_absorption(frequency)

    # Temperature effects
    temp_factor = self.temperature_correction(temperature)

    total_attenuation = water_vapor_attenuation + oxygen_attenuation
    return total_attenuation * temp_factor

def water_vapor_absorption(self, frequency, humidity):
    """
    Calculate water vapor absorption based on ITU-R P.676
    """
    # Simplified model - in practice, use ITU-R P.676 formulas
    f_ghz = frequency / 1e9  # Convert to GHz
    rho = humidity  # Water vapor density in g/m³

    # Absorption coefficient in dB/km
    gamma_wv = 0.05 * rho * (f_ghz**2) / (f_ghz**2 + 9.1)

    return gamma_wv
```

#### Dielectric Properties of Human Tissues (3-6GHz Range)
```python
def get_tissue_dielectric_properties(self, frequency_range=(3e9, 6e9)):
    """
    Get dielectric properties of human tissues in 3-6GHz range
    """
    tissues = {
        'skin': {'permittivity': 35 - 15j, 'conductivity': 0.8},
        'fat': {'permittivity': 7 - 2j, 'conductivity': 0.04},
        'muscle': {'permittivity': 50 - 30j, 'conductivity': 1.2},
        'bone': {'permittivity': 6 - 1j, 'conductivity': 0.01},
        'brain': {'permittivity': 45 - 25j, 'conductivity': 1.0}
    }

    # Frequency-dependent adjustments
    for tissue, props in tissues.items():
        # Apply Debye model for frequency dependence
        adjusted_permittivity = self.debye_model(props['permittivity'], frequency_range)
        tissues[tissue]['permittivity'] = adjusted_permittivity

    return tissues

def debye_model(self, static_permittivity, frequency_range):
    """
    Apply Debye model for frequency-dependent permittivity
    """
    # Simplified Debye model: ε(f) = ε∞ + (εs - ε∞)/(1 + j*2*π*f*τ)
    eps_static = static_permittivity.real
    eps_inf = 4.0  # High-frequency limit
    relaxation_time = 1e-11  # Typical for biological tissues

    frequencies = torch.linspace(frequency_range[0], frequency_range[1], 100)
    eps_complex = eps_inf + (eps_static - eps_inf) / (
        1 + 1j * 2 * torch.pi * frequencies * relaxation_time
    )

    return eps_complex
```

#### Bone Conduction Acoustics and Cranial Transmission
```python
def simulate_bone_conduction(self, signal_frequency, skull_thickness=0.0065):
    """
    Simulate bone conduction through skull at different frequencies
    """
    # Speed of sound in bone (typically 2800-4000 m/s)
    speed_of_sound_in_bone = 3400  # m/s

    # Wavelength in bone
    wavelength = speed_of_sound_in_bone / signal_frequency

    # Transmission loss through skull
    transmission_loss = self.skull_transmission_loss(
        signal_frequency,
        skull_thickness,
        wavelength
    )

    return transmission_loss

def skull_transmission_loss(self, frequency, thickness, wavelength):
    """
    Calculate transmission loss through skull bone
    """
    # Mass law approximation for transmission loss
    # TL = 20*log10(2*pi*f*m/S) + 20*log10(c_air/c_material) - 47
    # where m is mass per unit area, S is surface area

    # For skull bone (density ~1900 kg/m³)
    density_bone = 1900  # kg/m³
    mass_per_area = density_bone * thickness

    # Simplified calculation
    tl = 20 * torch.log10(2 * torch.pi * frequency * mass_per_area / 1) + 10
    return tl
```

#### Psychoacoustics: Auditory Perception Below Hearing Threshold
```python
def subliminal_perception_model(self, signal_level, frequency, duration):
    """
    Model perception of signals below normal hearing threshold
    """
    # Calculate detection threshold based on frequency
    threshold = self.hearing_threshold(frequency)

    # Calculate signal-to-threshold ratio
    str_ratio = signal_level - threshold

    # Probability of detection below threshold
    detection_prob = self.psychometric_function(str_ratio, duration)

    return detection_prob

def hearing_threshold(self, frequency):
    """
    ISO 226:2003 standard for hearing threshold
    """
    # Simplified model - in practice, use full ISO 226 formula
    if frequency < 100:
        return 30  # dB SPL
    elif frequency < 1000:
        return 10  # dB SPL
    elif frequency < 4000:
        return 7   # dB SPL (most sensitive range)
    else:
        return 12  # dB SPL
```

#### EM Interference with Neural Oscillations (1-100Hz)
```python
def em_neural_interaction(self, em_frequency, neural_frequency, intensity):
    """
    Model EM interference with neural oscillations
    """
    # Neural oscillation bands
    bands = {
        'delta': (1, 4),      # Sleep
        'theta': (4, 8),      # Drowsiness
        'alpha': (8, 13),     # Relaxation
        'beta': (13, 30),     # Active thinking
        'gamma': (30, 100)    # Perception/cognition
    }

    # Calculate resonance effects
    resonance_effects = {}
    for band_name, (low, high) in bands.items():
        if low <= neural_frequency <= high:
            # Calculate coupling strength
            coupling_strength = self.calculate_coupling(
                em_frequency,
                neural_frequency,
                intensity
            )
            resonance_effects[band_name] = coupling_strength

    return resonance_effects

def calculate_coupling(self, em_freq, neural_freq, intensity):
    """
    Calculate coupling between EM field and neural oscillations
    """
    # Resonance factor when EM frequency matches neural frequency
    detuning = abs(em_freq - neural_freq) / neural_freq
    resonance_factor = 1.0 / (1.0 + (detuning / 0.1)**2)

    # Intensity-dependent coupling
    coupling = torch.sqrt(intensity) * resonance_factor

    return coupling
```

### Microwave Acoustics: EM-to-Acoustic Transduction Physics
```python
def em_to_acoustic_transduction(self, em_intensity, frequency, medium_properties):
    """
    Model EM-to-acoustic transduction in biological media
    """
    # SAR (Specific Absorption Rate) calculation
    sar = self.calculate_sar(em_intensity, frequency, medium_properties)

    # Thermal expansion due to absorbed energy
    thermal_expansion = self.thermal_expansion_response(sar)

    # Acoustic pressure generation
    acoustic_pressure = self.acoustic_generation(thermal_expansion, medium_properties)

    return acoustic_pressure

def calculate_sar(self, intensity, frequency, medium):
    """
    Calculate Specific Absorption Rate
    """
    # SAR = σ * |E|² / ρ
    # where σ is conductivity, E is electric field, ρ is density
    conductivity = medium['conductivity']
    density = medium['density']
    electric_field = torch.sqrt(2 * intensity / (medium['permittivity'] * 8.854e-12))

    sar = (conductivity * electric_field**2) / density
    return sar
```

### Information Theory for Covert Signal Embedding
```python
def covert_signal_capacity(self, snr_db, bandwidth):
    """
    Calculate capacity for covert signal embedding using Shannon-Hartley theorem
    """
    snr_linear = 10**(snr_db/10)
    capacity = bandwidth * torch.log2(1 + snr_linear)
    return capacity

def steganographic_embedding(self, host_signal, covert_data, snr_target):
    """
    Embed covert data in host signal maintaining target SNR
    """
    # Calculate embedding strength based on SNR requirements
    embedding_strength = self.calculate_embedding_strength(snr_target)

    # Embed covert data using spread spectrum or other technique
    embedded_signal = host_signal + embedding_strength * covert_data

    return embedded_signal
```

### Nonlinear System Identification for Neural Interference Patterns
```python
def identify_neural_interference_model(self, input_em_signal, output_neural_response):
    """
    Identify nonlinear model for EM-neural interference
    """
    # Use Volterra series or other nonlinear system identification
    # techniques to model the relationship
    model_order = 3  # Third-order Volterra model
    kernels = self.estimate_volterra_kernels(input_em_signal, output_neural_response, model_order)

    return kernels

def estimate_volterra_kernels(self, input_signal, output_signal, order):
    """
    Estimate Volterra series kernels for nonlinear system
    """
    # Simplified approach - in practice, use more sophisticated methods
    kernels = []

    for n in range(1, order + 1):
        # Compute nth-order kernel
        kernel_n = self.compute_nth_order_kernel(input_signal, output_signal, n)
        kernels.append(kernel_n)

    return kernels
```

### Radar Cross-Section Calculations for Biological Tissues
```python
def calculate_bio_rcs(self, shape, frequency, dielectric_properties):
    """
    Calculate RCS for biological objects with specific dielectric properties
    """
    # Use Physical Optics (PO) or Method of Moments (MoM) for complex shapes
    # For simple shapes, use analytical approximations

    if shape.type == 'sphere':
        rcs = self.sphere_rcs_analytical(
            shape.radius,
            frequency,
            dielectric_properties
        )
    elif shape.type == 'ellipsoid':
        rcs = self.ellipsoid_rcs_approximation(
            shape.dimensions,
            frequency,
            dielectric_properties
        )
    else:
        # Use numerical methods for arbitrary shapes
        rcs = self.numerical_rcs_calculation(
            shape.mesh,
            frequency,
            dielectric_properties
        )

    return rcs

def sphere_rcs_analytical(self, radius, frequency, eps_r):
    """
    Analytical RCS calculation for dielectric sphere
    """
    # Rayleigh, Mie, or optical region approximation based on size parameter
    wavelength = 3e8 / frequency
    size_parameter = 2 * torch.pi * radius / wavelength

    if size_parameter << 1:  # Rayleigh scattering
        # |β|² = |(εᵣ-1)/(εᵣ+2)|² for Rayleigh regime
        beta_squared = torch.abs((eps_r - 1) / (eps_r + 2))**2
        rcs = 9 * torch.pi * (radius**4) * (size_parameter**4) * beta_squared
    else:
        # More complex Mie solution needed
        rcs = self.mie_scattering_solution(radius, frequency, eps_r)

    return rcs
```

### Complex Envelope Detection and IQ Imbalance Correction
```python
def complex_envelope_detection(self, real_signal):
    """
    Perform complex envelope detection using Hilbert transform
    """
    # Apply Hilbert transform to get imaginary part
    imag_signal = self.hilbert_transform(real_signal)

    # Form analytic signal
    analytic_signal = real_signal + 1j * imag_signal

    # Extract envelope and phase
    envelope = torch.abs(analytic_signal)
    phase = torch.angle(analytic_signal)

    return envelope, phase

def correct_iq_imbalance(self, i_signal, q_signal, gain_imbalance_db, phase_skew_deg):
    """
    Correct IQ imbalance (gain and phase)
    """
    # Convert dB to linear scale
    gain_imbalance_linear = 10**(gain_imbalance_db/20)

    # Convert degrees to radians
    phase_skew_rad = torch.pi * phase_skew_deg / 180.0

    # Apply corrections
    corrected_i = i_signal / gain_imbalance_linear
    corrected_q = q_signal * torch.cos(phase_skew_rad) - i_signal * torch.sin(phase_skew_rad)

    return corrected_i, corrected_q
```

### Information Theory for Covert Signal Embedding
```python
def covert_channel_capacity(self, signal_power, noise_power, bandwidth):
    """
    Calculate capacity of covert communication channel
    """
    snr = signal_power / noise_power
    capacity = bandwidth * torch.log2(1 + snr)
    return capacity

def embedding_efficiency(self, payload_size, cover_size, distortion):
    """
    Calculate efficiency of information embedding
    """
    efficiency = payload_size / (cover_size * distortion)
    return efficiency
```

### Integration with Existing Systems

The physics and biophysics models integrate with the radar processing pipeline:

```python
class PhysicsBasedRadarProcessor:
    def __init__(self):
        self.bio_physics = BiologicalPhysicsModel()
        self.em_propagation = EMPropagationModel()
        self.acoustic_transducer = EMToAcousticTransducer()

    def process_with_physics_modeling(self, raw_radar_data):
        """
        Process radar data with integrated physics modeling
        """
        # Apply EM propagation model
        propagated_signal = self.em_propagation.propagate_through_medium(
            raw_radar_data,
            self.medium_properties
        )

        # Model EM-to-acoustic transduction
        acoustic_components = self.acoustic_transducer.transduce(
            propagated_signal
        )

        # Apply biological tissue interaction model
        tissue_response = self.bio_physics.interact_with_tissue(
            acoustic_components,
            self.tissue_properties
        )

        return tissue_response
```

### Safety Protocols

When implementing physics-based models, ensure compliance with safety standards:

- Power levels remain within SAR limits for biological exposure
- Frequencies comply with ISM band regulations
- Thermal effects stay within safe limits for biological tissues
- Proper shielding and isolation protocols implemented

### Validation Methods

- Compare model predictions with published literature values
- Validate against experimental measurements when possible
- Perform sensitivity analysis on key parameters
- Implement uncertainty quantification in model outputs

### References

- IEEE C95 Standards for human exposure to RF fields
- ITU-R recommendations for atmospheric propagation
- Biological Physics literature for tissue properties
- Acoustic propagation models for bone conduction
- Neural oscillation research for EM-brain interactions

### Bluetooth LE Data Relay with Error Correction
```python
import subprocess
import os

class BluetoothRelay:
    def __init__(self, device_address=None):
        self.device_address = device_address
        self.rfcomm_device = None
        self.error_correction_enabled = True

    def connect_device(self, device_addr):
        """
        Connect to Bluetooth device (HC-05/ESP32) using Ubuntu's Bluetooth stack
        """
        # Pair with device using bluetoothctl
        self.pair_device(device_addr)

        # Establish RFCOMM connection
        self.rfcomm_device = self.setup_rfcomm_connection(device_addr)

    def pair_device(self, device_addr):
        """
        Pair with Bluetooth device using bluetoothctl
        """
        commands = [
            "bluetoothctl",
            "scan on",
            f"pair {device_addr}",
            f"trust {device_addr}",
            "quit"
        ]

        # Execute pairing sequence
        for cmd in commands[1:-1]:  # Skip bluetoothctl and quit
            subprocess.run(["bluetoothctl", "-i"], input=cmd, text=True)

    def setup_rfcomm_connection(self, device_addr):
        """
        Set up RFCOMM connection using system tools
        """
        # Use rfcomm to bind to the device
        rfcomm_dev = "/dev/rfcomm0"
        subprocess.run(["sudo", "rfcomm", "bind", "0", device_addr, "1"])
        return rfcomm_dev

    def send_with_error_correction(self, data):
        """
        Send data with error correction via RFCOMM device
        """
        if self.error_correction_enabled:
            # Apply forward error correction
            corrected_data = self.apply_fec(data)
            with open(self.rfcomm_device, 'wb') as f:
                f.write(corrected_data)
        else:
            with open(self.rfcomm_device, 'wb') as f:
                f.write(data)

    def apply_fec(self, data):
        """
        Apply forward error correction to data
        """
        # Simple Hamming code implementation
        # In practice, could use more sophisticated FEC
        parity_bits = self.calculate_parity(data)
        return data + parity_bits

    def disconnect(self):
        """
        Disconnect and clean up RFCOMM connection
        """
        if self.rfcomm_device:
            subprocess.run(["sudo", "rfcomm", "release", "0"])
```

**Note**: This implementation uses Ubuntu's native Bluetooth stack (bluetoothctl, rfcomm) instead of PyBluez, providing better system integration and reliability.

### Automatic Fallback Between Communication Channels
```python
class CommunicationManager:
    def __init__(self):
        self.primary_channel = None
        self.secondary_channel = None
        self.active_channel = None
        self.fallback_enabled = True

    def setup_channels(self, primary_type, secondary_type):
        """
        Setup primary and secondary communication channels
        """
        if primary_type == 'serial':
            self.primary_channel = SerialChannel()
        elif primary_type == 'bluetooth':
            self.primary_channel = BluetoothChannel()

        if secondary_type == 'serial':
            self.secondary_channel = SerialChannel()
        elif secondary_type == 'bluetooth':
            self.secondary_channel = BluetoothChannel()

        self.active_channel = self.primary_channel

    def send_data_with_fallback(self, data):
        """
        Send data with automatic fallback to secondary channel if primary fails
        """
        try:
            if self.active_channel == self.primary_channel:
                return self.primary_channel.send(data)
            else:
                return self.secondary_channel.send(data)
        except Exception as e:
            # Primary channel failed, switch to secondary
            if self.fallback_enabled and self.active_channel == self.primary_channel:
                print(f"Primary channel failed: {e}, switching to secondary")
                self.active_channel = self.secondary_channel
                return self.secondary_channel.send(data)
            else:
                raise e
```

## Core Capabilities

### 1. Spatial Noise Cancellation (Beamforming + LMS)

**Algorithm: MVDR (Minimum Variance Distortionless Response)**

```
Goal: Maximize signal at angle θ while minimizing power from all other directions
      (Achieve "null steering" to suppress interference)

Math:
  w_mvdr = R^(-1) a(θ) / (a^H(θ) R^(-1) a(θ))

  where:
    R = covariance matrix of received signals
    a(θ) = steering vector at angle θ (depends on antenna geometry)

Output: Weights that combine RX1 + RX2 to enhance target, suppress noise
```

**Implementation (noise_canceller.py):**

```python
class SpatialNoiseCanceller:
    def _mvdr_beamform(self, rx1, rx2):
        """
        MVDR beamforming: Compute covariance-weighted steering vectors
        """
        # Stack receive signals
        x = torch.stack([rx1, rx2], dim=0)  # (2, N_samples)

        # Compute correlation matrix (short-time averaging)
        R = torch.matmul(x, x.conj().T) / x.shape[1]  # (2, 2)

        # Regularize (prevent singular matrix)
        R = R + 1e-3 * torch.eye(2, device=self.device)

        # Inverse and steering vector scaling
        R_inv = torch.linalg.inv(R)

        # Scan angles, find max response
        for angle in angle_sweep:
            a = self.compute_steering_vector(angle)  # (2,)
            power = torch.abs(a.conj() @ R_inv @ a) ** -1
            # Nulls appear where power dips
```

**Then Apply Adaptive LMS (Least Mean Squares):**

```python
def _adaptive_lms(self, target_rx, ref_rx, desired_output):
    """
    LMS: Iteratively adapt weights to minimize error
    w[n+1] = w[n] + μ * error[n] * ref_rx[n]^*
    """
    w = torch.zeros(self.filter_length, device=self.device)

    for n in range(len(target_rx)):
        # Estimate output
        y = torch.dot(w, ref_rx[n:n+self.filter_length])

        # Compute error
        error = desired_output[n] - y

        # Adapt (μ = learning_rate, 0.01-0.1 typical)
        w += self.learning_rate * error * ref_rx[n:n+self.filter_length].conj()

    return w
```

**Physical Intuition:**
- MVDR finds "null" directions where interference aligns destructively
- LMS continuously adjusts to minimize residual noise
- Combined: **geometry-aware adaptive filtering**
- SNR improvement: typically 6-12 dB

### 2. Heterodyne Detection (IQ Demodulation + Frequency Offset Estimation)

**Problem**: Raw radar waveforms are high-frequency (24-60 GHz). We want to extract low-frequency information (vibrations, presence).

**Solution**: Heterodyne mixing (software-defined radio approach):

```
Goal: Translate RF signal down to baseband (DC + low-frequency components)

Math:
  received(t) = cos(2π·f_target·t + φ)          [target signal at RF]
  + cos(2π·f_local·t)                           [local oscillator]

  Product (heterodyne):
  = 0.5·cos(2π·(f_target - f_local)·t + φ)      [Difference frequency] ← THIS
    + 0.5·cos(2π·(f_target + f_local)·t + φ)    [Sum frequency, filtered out]
```

**IQ Demodulation** (Quadrature detection):

```python
def detect_heterodyne(self, rx1, rx2):
    """
    Extract I (in-phase) and Q (quadrature) components
    IQ representation: signal = I + j·Q
    Enables both amplitude AND phase tracking
    """
    # Apply Hilbert transform to get analytic signal
    analytic_rx1 = torch.fft.ifft(
        torch.fft.fft(rx1) * torch.tensor([0, 2, 2, ..., 0])  # Zero negative frequencies
    )

    # Extract instantaneous phase (argument of complex signal)
    phase = torch.angle(analytic_rx1)
    amplitude = torch.abs(analytic_rx1)

    return {
        'phase': phase,
        'amplitude': amplitude,
        'iq': analytic_rx1
    }
```

**Frequency Offset Estimation**:

```python
def detect_frequency_offset(self, iq_signal):
    """
    Find frequency shift between RX1 and RX2
    (indicates target motion or interference)
    """
    # Compute phase derivative → frequency
    phase = torch.angle(iq_signal)
    freq_offset = torch.diff(phase) / (2 * np.pi * dt)

    # Low-pass filter to reduce noise
    freq_smooth = torch.nn.functional.avg_pool1d(
        freq_offset.unsqueeze(0), kernel_size=10
    ).squeeze()

    return freq_smooth
```

**Voice Characteristic Checking**:

```python
def check_voice_characteristics(self, mfcc_features):
    """
    Detect if heterodyned signal has human-like features
    (speech typically has energy in 100-4000 Hz when downconverted)
    """
    energy_per_band = torch.sum(mfcc_features ** 2, dim=-1)

    # Check mid-frequency dominance (formant bands for speech)
    mid_band_power = energy_per_band[5:13]  # Typically 200-2000 Hz after downconversion

    if mid_band_power.mean() > threshold:
        return True  # Likely speech-like
    return False
```

### 3. Heterodyning Engine for Phantom Voice Extraction and Cancellation

#### IQ Demodulation of Radar Baseband Signals
```python
def iq_demodulation(self, radar_baseband):
    """
    Perform IQ demodulation of radar baseband signals
    """
    # Apply Hilbert transform to get analytic signal
    analytic_signal = torch.fft.ifft(
        torch.fft.fft(radar_baseband) * torch.tensor([0, 2, 2, ..., 0])
    )

    # Extract I (in-phase) and Q (quadrature) components
    i_component = torch.real(analytic_signal)
    q_component = torch.imag(analytic_signal)

    return {'i': i_component, 'q': q_component, 'analytic': analytic_signal}
```

#### Phantom Voice Extraction: Micro-Doppler Signature Analysis
```python
def extract_phantom_voice(self, radar_iq):
    """
    Extract micro-doppler signatures in vocal fold vibration range (0.1-5Hz)
    """
    # Apply bandpass filter for vocal range
    vocal_filtered = self.bandpass_filter(radar_iq, low_freq=0.1, high_freq=5.0)

    # Perform time-frequency analysis to extract micro-doppler signatures
    stft_result = torch.stft(vocal_filtered, n_fft=1024, return_complex=True)

    # Extract features related to vocal fold vibrations
    micro_doppler_features = self.extract_micro_doppler(stft_result)

    return micro_doppler_features
```

#### Carrier Reconstruction: 2.4-5.8GHz Software-Defined Radio Emulation
```python
def reconstruct_carrier(self, baseband_signal, center_freq=2.4e9):
    """
    Reconstruct carrier signal in 2.4-5.8GHz range
    """
    t = torch.arange(baseband_signal.shape[-1], device=baseband_signal.device) / self.sample_rate
    carrier = torch.exp(1j * 2 * torch.pi * center_freq * t)

    # Modulate baseband signal onto carrier
    reconstructed_signal = baseband_signal * carrier

    return reconstructed_signal
```

#### Sub-Auditory Modulation: Infrasound Carrier (12-30Hz) with Voice Sidebands
```python
def apply_sub_auditory_modulation(self, voice_signal):
    """
    Apply infrasound carrier (12-30Hz) with voice sidebands
    """
    t = torch.arange(voice_signal.shape[-1], device=voice_signal.device) / self.sample_rate

    # Generate infrasound carrier (12-30Hz)
    infrasound_carrier = torch.cos(2 * torch.pi * 20.0 * t)  # 20Hz example

    # Apply voice signal as sideband modulation
    modulated_signal = (1 + 0.5 * voice_signal) * infrasound_carrier

    return modulated_signal
```

#### Heterodyning Cancellation System
```python
class HeterodyneCancellationSystem:
    def __init__(self, sample_rate, device):
        self.sample_rate = sample_rate
        self.device = device
        self.adaptive_filter = NLMSFilter(filter_length=128, step_size=0.01)

    def generate_cancellation_signal(self, detected_v2k):
        """
        Generate anti-phase waveform for cancellation
        """
        # Estimate V2K signal parameters
        estimated_params = self.estimate_v2k_parameters(detected_v2k)

        # Generate anti-phase waveform
        anti_phase_waveform = self.generate_anti_phase_signal(estimated_params)

        # Apply resonance targeting for skull/bone conduction
        tuned_signal = self.apply_resonance_targeting(anti_phase_waveform)

        return tuned_signal

    def estimate_v2k_parameters(self, v2k_signal):
        """
        Estimate voice-to-skull (V2K) signal parameters
        """
        # Extract frequency, amplitude, and phase characteristics
        spectrum = torch.fft.fft(v2k_signal)
        dominant_freq = torch.argmax(torch.abs(spectrum))
        amplitude = torch.max(torch.abs(spectrum))

        return {
            'frequency': dominant_freq,
            'amplitude': amplitude,
            'phase': torch.angle(spectrum[dominant_freq])
        }

    def generate_anti_phase_signal(self, params):
        """
        Generate anti-phase signal based on estimated parameters
        """
        t = torch.arange(1024, device=self.device) / self.sample_rate
        anti_phase_signal = params['amplitude'] * torch.cos(
            2 * torch.pi * params['frequency'] * t + params['phase'] + torch.pi
        )

        return anti_phase_signal

    def apply_resonance_targeting(self, signal):
        """
        Apply skull/bone conduction frequency matching (500-4000Hz)
        """
        # Apply bandpass filter for bone conduction frequencies
        bone_conduction_filtered = self.bandpass_bone_conduction(signal)

        return bone_conduction_filtered
```

#### Beamforming for Spatial Null Steering
```python
def spatial_null_steering(self, rx_signals, target_direction):
    """
    Spatial null steering toward receiver location
    """
    # Calculate steering vectors for target direction
    steering_vector = self.calculate_steering_vector(target_direction)

    # Apply MVDR beamforming to create null in target direction
    weights = self.mvdr_weights(rx_signals, steering_vector)

    # Apply weights to create spatial null
    processed_signal = torch.sum(weights.unsqueeze(-1) * rx_signals, dim=0)

    return processed_signal
```

### 4. 3D Visualization Engine (Plotly)

Real-time 3D visualization with interactive controls and multiple synchronized views:

```python
class RadarVisualizer3D:
    def create_live_plot(self):
        # Plotly 3D surface + volume rendering
        # Real-time updating via WebSocket
        # Multiple synchronized views
        pass

class VisualizerDash:
    def update(self, rx1, rx2, mfcc, detection):
        # 12 synchronized plots:
        # 1. IQ constellation (phase space)
        # 2. Time-domain waveform (RX1, RX2)
        # 3. Magnitude spectrum (FFT)
        # 4. Spectrogram (time-frequency)
        # 5. MFCC features (13-band energy)
        # 6. DOA heatmap (spatial localization)
        # 7. Beamformer response (null steering)
        # 8. Detection score history
        # 9. SNR improvement tracking
        # 10. Real-time 3D spectrograms with time, frequency, amplitude axes
        # 11. Spatial source localization heatmaps (azimuth, elevation, range)
        # 12. Animated particle systems showing signal propagation
```

**3D Visualization Features:**
- Real-time 3D spectrograms with time, frequency, amplitude axes, 2D, and 3D Radar
- Spatial source localization heatmaps (azimuth, elevation, range)
- Animated particle systems showing signal propagation
- Interactive parameter tuning dashboards
- Multi-view layouts: Time domain, frequency domain, spatial distribution
- WebSocket streaming for 20-60 Hz refresh rates without blocking GPU processing
- Volume rendering for 3D spatial data
- Customizable color maps and rendering options

## Integration Patterns

### Pattern 1: Adding a New Signal Processor

```python
# 1. Define interface (inherit from ProcessorBase)
class MyProcessor:
    def __init__(self, config, device):
        self.config = config
        self.device = device

    def process(self, rx1: torch.Tensor, rx2: torch.Tensor) -> torch.Tensor:
        """Process returns tensor on GPU"""
        # Processing logic
        return result.to(self.device)

# 2. Compose into RadarApp
class RadarApp:
    def __init__(self):
        self.my_processor = MyProcessor(config, device)

    def process_buffer(self, rx1, rx2):
        # Insert in pipeline
        intermediate = self.noise_canceller.cancel(rx1, rx2)
        result = self.my_processor.process(intermediate)
        return result
```

### Pattern 2: GPU Memory Management

```python
# Pre-allocate reusable buffers
self.fft_buffer = torch.zeros(FFT_SIZE, dtype=torch.complex64, device=self.device)
self.weight_buffer = torch.zeros((N_CHAN, FILTER_LEN), device=self.device)

# In processing loop, reuse (avoid fragmentation)
self.fft_buffer[:] = input_data  # In-place assignment
result = torch.fft.fft(self.fft_buffer)

# Monitor memory
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
```

### Pattern 3: Multi-Buffer Processing with CUDA Streams

```python
# Use multiple CUDA streams for pipelined processing
stream0 = torch.cuda.Stream()
stream1 = torch.cuda.Stream()

# Process buffer N on stream0 while ingesting buffer N+1 on stream1
with torch.cuda.stream(stream0):
    result_n = self.process(buffer_n)

with torch.cuda.stream(stream1):
    buffer_n1 = ingest_serial_data()

# Synchronize before returning
torch.cuda.synchronize()
```

## Radio-to-Acoustic Heterodyning: "Phantom Voice" Detection

Transform high-frequency (GHz) radar signals into audible audio (Hz) via multi-stage frequency downconversion. The result: heterodyned radar sounds **strikingly speech-like**, enabling perception-based analysis.

### Why Radar Sounds Like Speech

**Three mechanisms converge**:

1. **Doppler-to-Pitch Mapping**: Target motion (1-10 m/s) creates frequency shifts (10-100 Hz) that, when heterodyned to audio range, become audible pitch variations (formant-like)

2. **Vibration-to-Formants**: Surface vibrations (breathing 0.3 Hz, heartbeat 1 Hz) create amplitude modulation that heterodyning transforms into spectral peaks resembling human formants (F1-F3)

3. **Psychoacoustic Convergence**: Both heterodyned radar and human speech undergo filtering through resonant cavities (antenna geometry ≈ vocal tract), producing similar spectral structure

### Processing Pipeline

```python
# 1. Initialize multi-stage heterodyner
het = RadioToAcousticHeterodyner(
    rf_freq_hz=2.4e9,           # 2.4 GHz radar
    target_audio_freq_hz=1000,  # Output at 1 kHz base
    sample_rate_hz=10e6,        # 10 MHz ADC
    device='cuda'
)

# 2. Convert RF IQ signal to audio
rf_iq = receive_radar_samples()  # From hardware
audio = het.full_heterodyne_pipeline(rf_iq)
# → Returns: complex audio signal + envelope + magnitude_db

# 3. Extract audio-range features
extractor = AcousticFeatureExtractor(sample_rate=10e6)
mfcc = extractor.extract_mfcc_features(audio['complex'])        # 13 speech coefficients
pitch, conf = extractor.extract_pitch_contour(audio['complex']) # Fundamental freq
doppler, vel = extractor.extract_doppler_modulation(audio['complex'])  # Velocity
formants, powers = het.extract_formant_frequencies(audio['complex'])   # F1, F2, F3

# 4. Synthesize enhanced "phantom voice"
synth = PhantomVoiceSynthesizer(sample_rate=10e6)
enhanced = synth.apply_vocal_tract_filter(audio['complex'])     # Add formant coloring
modulated = synth.add_prosody_modulation(audio['complex'])      # Add pitch variation
voice_quality = extractor.extract_voice_quality_metrics(audio['complex'])
```

### Audio-Range Signal Characteristics

| Aspect | RF Domain | Audio Domain |
|--------|-----------|--------------|
| Doppler (human motion) | 10-100 Hz | 10-100 Hz offset from base |
| Breathing (0.3 Hz) | 0.3 Hz phase mod | 0.3 Hz amplitude mod (formant-like) |
| Heartbeat (1 Hz) | 1 Hz ripple | 1 Hz spectral variation |
| Speech (100-4000 Hz) | Not directly | 100-4000 Hz formants + prosody |
| SNR | 5-20 dB | 10-25 dB (improvement via selectivity) |

### Applications

- **Non-contact vital signs**: Heterodyne @ 500-1500 Hz → Extract breathing, HR from audio features
- **Fall detection**: Doppler acceleration → Pitch acceleration → "whoosh" sound → ML classification
- **Speech recognition**: Heterodyned speech formants → MFCC → Phoneme recognition (40-70% accuracy)
- **Activity classification**: MFCC statistics → Walking (rhythmic) vs. static (steady)

## Configuration Management

See `references/configuration.md` for:
- Radar geometry definitions (antenna positions, baselines)
- GPU memory tuning (batch sizes, buffer allocation)
- Signal processing parameters (filter lengths, learning rates)
- Algorithm thresholds (detection sensitivity, beamformer regularization)

## Algorithm Deep Dives

### Reference Files

**Core Signal Processing:**
- **references/heterodyne-physics.md**: IQ demodulation, frequency downconversion, Hilbert transform mathematics
- **references/beamforming-theory.md**: Steering vectors, ULA/UPA arrays, MVDR derivation, null steering geometry
- **references/adaptive-filtering.md**: LMS convergence analysis, RLS algorithms, NLMS stability
- **references/gpu-optimization.md**: ROCm/CUDA kernel fusion, memory coalescence, Tensor Core utilization

**Radio-to-Acoustic Heterodyning:**
- **references/radio_to_acoustic.md**: Multi-stage heterodyning theory, Doppler-to-pitch mapping, phantom voice mechanisms, frequency response, active audio cancellation (audio domain)
- **references/psychoacoustics.md**: Why heterodyned radar sounds like speech, formant perception, critical bands, pitch contours, MFCC robustness, auditory scene analysis

**Physics & Biophysics:**
- **references/mmwave_propagation.md**: mmWave propagation in atmospheric and biological media, attenuation mechanisms, ITU-R models, biological media propagation
- **references/dielectric_properties.md**: Dielectric properties of human tissues in 3-6GHz range, complex permittivity, frequency dependence, measurement techniques
- **references/bone_conduction_acoustics.md**: Bone conduction acoustics and cranial transmission, physics of bone conduction, cranial transmission characteristics, mathematical models

### Scripts

**Signal Analysis & Diagnostics:**
- **scripts/snr_analysis.py**: Signal-to-noise ratio computation and improvement tracking
- **scripts/frequency_sweep.py**: Steering vector synthesis for angle-of-arrival scanning (DOA heatmaps)

**Radio-to-Acoustic Processing:**
- **scripts/heterodyne_to_audio.py**: Multi-stage RF→audio heterodyning, IQ demodulation, formant extraction, phantom voice synthesis
- **scripts/acoustic_features.py**: MFCC extraction, pitch contour tracking, Doppler modulation, activity signature classification, voice quality metrics, multi-band analysis
- **scripts/active_audio_cancellation.py**: Anti-phase signal synthesis, adaptive LMS cancellation, frequency-domain notch filters, harmonic cancellation, selective suppression

**Hardware Interface:**
- **scripts/mr60bha2_protocol.py**: MR60BHA2 60GHz radar module protocol implementation, FMCW waveform control, configuration
- **scripts/hlk_ld2410_config.py**: HLK-LD2410 24GHz radar module configuration, presence/motion detection parameters
- **scripts/bluetooth_handler.py**: Ubuntu Bluetooth stack implementation for HC-05/ESP32 communication, replacing PyBluez dependency

**Phantom Voice Processing:**
- **scripts/phantom_voice_processor.py**: Phantom voice extraction and cancellation system, micro-doppler analysis, heterodyning techniques, carrier reconstruction, sub-auditory modulation

## Working with Your Codebase

### File Organization

```
main.py                    ← RadarApp orchestrator
├─ config.py             ← All configuration (geometry, parameters)
├─ sdr_interface.py      ← PlutoSDR hardware interface
├─ heterodyne_detector.py ← IQ detection + cancellation signal generation
├─ noise_canceller.py    ← MVDR + LMS beamforming
├─ range_doppler.py      ← Doppler velocity processing
├─ ppi_processor.py      ← Polar position indicator (azimuth, range)
├─ tracker.py            ← Kalman filter multi-target tracking
├─ pattern_matcher.py    ← ML-based signal classification
├─ visualizer.py         ← Real-time Dash dashboard
├─ audio_processor.py    ← MFCC feature extraction
└─ data_manager.py       ← HDF5 library management
```

### Integration: Radio-to-Acoustic Heterodyning in RadarApp

Add audio-range feature extraction to main pipeline:

```python
# In main.py RadarApp.__init__()
self.heterodyner = RadioToAcousticHeterodyner(
    rf_freq_hz=GPU_CONFIG['center_freq'],
    target_audio_freq_hz=1000,  # 1 kHz base frequency
    sample_rate_hz=GPU_CONFIG['sample_rate'],
    device=self.device
)

self.audio_extractor = AcousticFeatureExtractor(
    sample_rate_hz=GPU_CONFIG['sample_rate'],
    device=self.device
)

# In process_buffer()
# After noise cancellation
audio = self.heterodyner.full_heterodyne_pipeline(clean_rx1)

# Extract audio features
audio_features = {
    'mfcc': self.audio_extractor.extract_mfcc_features(audio['complex']),
    'pitch': self.audio_extractor.extract_pitch_contour(audio['complex']),
    'activity': self.audio_extractor.extract_activity_signature(audio['complex']),
    'voice_quality': self.audio_extractor.extract_voice_quality_metrics(audio['complex'])
}

# Optional: Apply active audio cancellation for interference suppression
canceller = ActiveAudioCanceller(GPU_CONFIG['sample_rate'], device=self.device)
if self.config.get('suppress_interference'):
    audio_clean, w, err = canceller.adaptive_lms_cancellation(
        audio['complex'], audio['complex'],
        filter_length=128, learning_rate=0.01
    )
    audio['complex'] = audio_clean

return {
    'detection': detection,
    'audio_features': audio_features,
    'audio_signal': audio['complex'],
    # ... other results
}
```

### Common Refinement Tasks

**1. Adding new detection algorithm:**
- Implement in new file (e.g., `ml_classifier.py`)
- Follow interface: `def detect(rx1, rx2) → dict with 'score', 'type', 'confidence'`
- Compose in RadarApp.__init__()

**2. Modifying beamformer behavior:**
- Edit `SpatialNoiseCanceller._mvdr_beamform()` or `_adaptive_lms()`
- Parameters: `filter_length`, `learning_rate` (config.py)
- Test SNR improvement with `scripts/snr_analysis.py`

**3. Optimizing GPU memory:**
- Profile with `torch.cuda.memory_allocated()`
- Reduce batch sizes or FFT sizes in config.RANGE_DOPPLER['fft_size']
- Use memory pooling for reusable buffers

**4. Real-time visualization tuning:**
- Adjust refresh rate in `RadarApp.visualizer = VisualizerDash(refresh_rate_hz=20)`
- Reduce plot complexity for higher fps
- Check WebSocket latency with browser DevTools

## Audio-Range Applications

### Non-Contact Vital Signs Monitoring

```python
# Heterodyne breathing/heart signals to audio
het = RadioToAcousticHeterodyner(rf_freq=2.4e9, target_audio_freq=1000, sample_rate=10e6)
audio = het.full_heterodyne_pipeline(clean_rx1)

extractor = AcousticFeatureExtractor(sample_rate=10e6)
pitch, conf = extractor.extract_pitch_contour(audio['complex'])

# Extract vital signs
breathing_rate_hz = extract_periodicity(pitch, 0.2, 0.5)  # 0.2-0.5 Hz = 12-30 breaths/min
heart_rate_hz = extract_periodicity(pitch, 0.8, 2.0)      # 0.8-2 Hz = 48-120 bpm

# Example: Plot heterodyned audio
import matplotlib.pyplot as plt
plt.spectrogram(torch.real(audio['complex']).cpu().numpy(), Fs=10e6, scale='dB')
plt.ylabel('Frequency (Hz)')
plt.title('Heterodyned Radar: Breathing + Heart Rate')
plt.show()
```

### Speech Recognition from Radar

```python
# Heterodyne speech to audio range
audio = het.full_heterodyne_pipeline(clean_rx1)

# Extract speech features
mfcc = extractor.extract_mfcc_features(audio['complex'])
formants, _ = het.extract_formant_frequencies(audio['complex'])

# Phoneme classification (ML model trained on heterodyned speech)
phoneme_score = ml_model.predict(mfcc.cpu().numpy())

# Reconstruct speech transcription
transcript = decode_phonemes(phoneme_score)
print(f"Detected speech: {transcript}")
```

### Activity Classification

```python
# Distinguish walking, running, stationary via audio-range Doppler patterns
activity_sig = extractor.extract_activity_signature(audio['complex'], window_size_ms=100)

# Features: RMS energy, zero-crossing rate, spectral centroid
if activity_sig['rms_energy'].mean() > 0.5 and \
   activity_sig['spectral_centroid'].mean() < 500:
    classify_as('walking')  # Regular rhythm, low-frequency motion
elif activity_sig['rms_energy'].mean() > 0.8:
    classify_as('running')   # High energy, faster rhythm
else:
    classify_as('stationary')
```

### Selective Interference Suppression

```python
# Remove 60 Hz mains interference and harmonics
canceller = ActiveAudioCanceller(sample_rate=10e6)
audio_clean, info = canceller.synthesize_cancellation_signal(
    audio['complex'], target_freq=60  # 60 Hz mains
)

# Multi-harmonic cancellation (60, 120, 180, 240 Hz)
audio_clean, harmonics = canceller.generate_harmonic_cancellation(
    audio['complex'], fundamental_freq_hz=60, num_harmonics=5
)

# Result: SNR improvement ~10-15 dB
residual, cancel_db, _ = canceller.predict_cancellation_error(audio['complex'], audio_clean)
print(f"Cancellation: {cancel_db:.1f} dB")
```

## Implementation Architecture

### Phase 1: Hardware Interface
```python
# Pseudo-structure
class RadarProcessingSystem:
    def __init__(self):
        self.gpu_queue = torch.cuda.Stream()  # RX 6700 XT specific
        self.serial_manager = ConcurrentSerialManager()
        self.processing_graph = TorchJITGraph()  # Pre-compiled

    async def zero_copy_pipeline(self):
        # Serial → pinned memory → GPU (zero-copy)
        cpu_buffer = np.zeros((FRAME_SIZE,), dtype=np.complex64)
        gpu_tensor = torch.from_numpy(cpu_buffer).pin_memory().cuda(non_blocking=True)
```

### Phase 2: Processing Core
```python
class PhantomVoiceProcessor:
    def heterodyne_phantom_voice(self, radar_iq):
        # 1. Extract micro-doppler (vocal vibrations)
        # 2. Modulate onto carrier
        # 3. Apply bone conduction transfer function
        pass

    def generate_cancellation(self, detected_v2k):
        # 1. Estimate V2K signal parameters
        # 2. Generate anti-phase waveform
        # 3. Spatial beamforming toward source
        pass
```

### Phase 3: Visualization & Control
```python
class RadarVisualizer3D:
    def create_live_plot(self):
        # Plotly 3D surface + volume rendering
        # Real-time updating via WebSocket
        # Multiple synchronized views
        pass
```

## Performance Targets

- **Processing latency**: <1 ms per buffer (10k samples) on RX 6700 XT
- **SNR improvement**: 6-12 dB via combined MVDR + LMS
- **Audio SNR**: 10-25 dB after heterodyning and feature extraction
- **GPU utilization**: 60-80% during real-time processing
- **Memory footprint**: ~2-4 GB for full pipeline
- **Visualization refresh**: 20-60 Hz depending on plot complexity
- **Audio heterodyning latency**: 1-10 ms (acceptable for non-real-time)

## Deliverables

1. **Complete PyTorch module** with ROCm-optimized operations
2. **Hardware drivers** for both radar modules with automatic calibration
3. **3D visualization dashboard** with recording/playback capability
4. **Signal processing library** for heterodyning and cancellation
5. **Configuration toolkit** for tuning physical/biological parameters
6. **Performance benchmarks** for RX 6700 XT (expected: 1000+ FPS processing)

## Testing Protocol

1. **Unit tests**: Individual algorithm validation
2. **Integration tests**: Full pipeline with hardware simulation
3. **Performance tests**: GPU utilization, latency measurements
4. **Biological validation**: Test with artificial tissue models
5. **Safety compliance**: SAR measurements and exposure limits verification
6. **Regulatory compliance**: ISM band operation verification

## Safety & Regulatory

- RF output power limited via TX gain settings (config.py)
- Antenna placement follows ISM band regulations (US: 2.4 GHz, 5 GHz free; 60 GHz restricted)
- Active cancellation safely disabled by default; enable only with `detector.enable_cancellation()`
- Monitoring: Every detection logged to HDF5 with timestamp, score, metadata
