# Heterodyne Detection Physics & Mathematics

## Core Concept: Frequency Translation via Mixing

**Problem**: Radar signals are at GHz frequencies (24-60 GHz). We can't directly measure or process them efficiently.

**Solution**: Mix with a local oscillator to translate to baseband (near DC + low frequencies).

### Mathematical Foundation

**Received Signal** (from target at range R, velocity v):
```
s(t) = A·cos(2π·f_RF·t + 2·(4π·R/λ) + 2π·(2·v/λ)·t + φ₀)
       └─────────────────────────────────────────────────────┘
              Constant              Range phase    Doppler shift
```

Where:
- f_RF: RF carrier frequency (e.g., 2.4 GHz)
- λ: Wavelength = c / f_RF
- Range phase: 4πR/λ (round-trip distance in wavelengths)
- Doppler shift: 2v/λ (positive for approaching targets)

**Local Oscillator** (generated locally):
```
LO(t) = cos(2π·f_LO·t)

Typical: f_LO = f_RF (or very close)
```

**Mixing (Multiplication)**:
```
s(t) × LO(t) = A·cos(2π·f_RF·t + φ_signal) × cos(2π·f_LO·t)

Using: cos(A)·cos(B) = ½[cos(A+B) + cos(A-B)]

Result = ½·A·cos(2π·(f_RF+f_LO)·t + φ) + ½·A·cos(2π·(f_RF-f_LO)·t + φ)
         └─ High frequency (RF+LO), filtered out ─┘   └─ Baseband, KEEP ─┘
```

**After Low-Pass Filter** (keeps only f_RF - f_LO):
```
Baseband_signal(t) = ½·A·cos(2π·(f_RF-f_LO)·t + 2·(4π·R/λ) + 2π·(2·v/λ)·t + φ₀)

If f_RF = f_LO: Baseband_signal(t) ≈ ½·A·[constant phase + Doppler shift]
```

## In-Phase (I) and Quadrature (Q) Representation

**Goal**: Capture both amplitude AND phase information.

### Why IQ?

Real-valued signals have ambiguity:
```
s(t) = A·cos(ωt + φ)  ← Can't distinguish if φ is changing or A is changing
```

Complex representation eliminates ambiguity:
```
s_complex(t) = A·e^(j(ωt + φ)) = A·e^(jφ) · e^(jωt)
                                   └─ Carrier     └─ Phase evolution
```

### IQ Demodulation Process

**Step 1**: Create two reference signals (90° apart):
```
I_ref(t) = cos(2π·f_IF·t)     [In-phase]
Q_ref(t) = sin(2π·f_IF·t)     [Quadrature, 90° shifted]
```

**Step 2**: Mix signal with both:
```
I(t) = s(t) × I_ref(t) = A·cos(φ(t))    [Amplitude modulation]
Q(t) = s(t) × Q_ref(t) = A·sin(φ(t))    [Phase modulation]
```

**Step 3**: Form complex signal:
```
Complex(t) = I(t) + j·Q(t) = A·e^(jφ(t))
```

### PyTorch Implementation

**Using Hilbert Transform** (analytic signal):
```python
def iq_demodulate(signal, sample_rate, carrier_freq):
    """
    Extract IQ components via Hilbert transform

    signal: real-valued input (from ADC)
    carrier_freq: Expected RF or intermediate frequency
    """
    # Hilbert transform: x_analytic = x(t) + j·H[x(t)]
    # Where H[] is the Hilbert transform (90° phase shifter)

    fft_signal = torch.fft.fft(signal)
    N = len(signal)

    # Zero out negative frequencies (imag part of Hilbert is zero there)
    h = torch.zeros_like(fft_signal)
    h[1:N//2] = 2  # Positive frequencies get factor of 2
    h[N//2] = 1 if N % 2 == 0 else 2  # Nyquist frequency

    analytic = torch.fft.ifft(fft_signal * h)

    # Extract IQ
    I = torch.real(analytic)
    Q = torch.imag(analytic)

    return I, Q

# Alternative: Direct IQ via complex mixing
def iq_demodulate_mixer(signal, sample_rate, carrier_freq):
    """Mix with complex exponential"""
    t = torch.arange(len(signal), dtype=torch.float32) / sample_rate
    complex_carrier = torch.exp(-2j * torch.pi * carrier_freq * t)
    iq_signal = signal * complex_carrier
    return torch.real(iq_signal), torch.imag(iq_signal)
```

## Phase Evolution and Frequency Detection

### Phase-Based Doppler Estimation

Instantaneous phase reveals Doppler shift:

```
φ(t) = 2π·(2·v/λ)·t + constant

Instantaneous frequency: f_inst = (1/2π)·dφ/dt = 2·v/λ

Therefore: v = (λ/2)·f_inst
```

**PyTorch Implementation**:
```python
def estimate_doppler_velocity(iq_signal, wavelength, sample_rate):
    """
    Estimate velocity from phase trajectory
    """
    # Unwrap phase to handle discontinuities
    phase = torch.angle(iq_signal)
    phase_unwrapped = torch.cumsum(
        torch.where(torch.diff(phase) > np.pi,
                   torch.diff(phase) - 2*np.pi,
                   torch.diff(phase)),
        dim=0
    )

    # Fit line to phase (least squares)
    t = torch.arange(len(phase_unwrapped)) / sample_rate
    # phase_unwrapped = slope·t + intercept
    slope = torch.linalg.lstsq(
        torch.stack([t, torch.ones_like(t)], dim=-1),
        phase_unwrapped
    ).solution[0]

    # Recover velocity
    freq_doppler = slope / (2 * np.pi)
    velocity = (wavelength / 2) * freq_doppler

    return velocity
```

## Ambiguity Function & Resolution

### Range-Doppler Ambiguity

Radar cannot distinguish:
1. Different **ranges** (if bandwidth is too narrow)
2. Different **velocities** (if observation time is too short)

**Range Resolution**:
```
Δr = c / (2·BW)

Example: 10 MHz bandwidth
Δr = 3×10⁸ / (2 × 10×10⁶) = 15 meters
```

**Velocity Resolution**:
```
Δv = λ / (2·T)

Where T = observation time

Example: λ=0.125 m, T=100 ms
Δv = 0.125 / (2 × 0.1) = 0.625 m/s
```

### Joint Resolution via 2D FFT

**Range-Doppler map**:
```python
def range_doppler_map(iq_signal, num_chirps, samples_per_chirp,
                      sample_rate, center_freq):
    """
    Create 2D range-Doppler image

    Dimension 1 (Range): FFT across samples within chirp
    Dimension 2 (Doppler): FFT across chirps
    """
    c = 299792458.0
    wavelength = c / center_freq

    # Reshape: (num_chirps, samples_per_chirp)
    matrix = iq_signal.reshape(num_chirps, samples_per_chirp)

    # FFT on range (axis 1)
    range_fft = torch.fft.fft(matrix, dim=1)

    # FFT on Doppler (axis 0)
    doppler_fft = torch.fft.fft(range_fft, dim=0)

    # Magnitude gives power
    power_map = torch.abs(doppler_fft) ** 2

    # Convert axes to physical units
    ranges = (torch.arange(samples_per_chirp) * c / sample_rate / 2)[:samples_per_chirp//2]
    velocities = (torch.arange(num_chirps) * wavelength / 2 / T)[:num_chirps//2]

    return power_map, ranges, velocities
```

## Practical Issues & Solutions

### Phase Ambiguity (Aliasing)

**Problem**: Phase only known mod 2π. Wraps every λ/2 of range change.

```
λ/2 for 2.4 GHz = 0.0625 m = 6.25 cm

Velocity ambiguity: ±v_max = ±λ/(2·Tc)

Where Tc = chirp time (typically 1-100 ms)
```

**Solution**: Unwrapping (track phase continuously):
```python
def unwrap_phase(phase):
    """Unwrap discontinuous phase"""
    dphase = torch.diff(phase)
    # Detect jumps > π
    jumps = torch.where(torch.abs(dphase) > np.pi)
    # Add 2π offset
    corrections = torch.cumsum(
        torch.where(dphase > np.pi, -2*np.pi,
                   torch.where(dphase < -np.pi, 2*np.pi, 0))
    )
    return torch.cat([phase[0:1], phase[1:] + corrections])
```

### IQ Imbalance (Hardware Imperfection)

Real hardware has amplitude & phase mismatch between I and Q channels:

```
Measured: I'(t) = A_I · I(t) + offset_I
          Q'(t) = A_Q · Q(t) + offset_Q + phase_error

Error: ΔA = A_I - A_Q, Δφ = phase_error
```

**Correction via Calibration**:
```python
def iq_balance_correction(I, Q, IQ_imbalance_matrix):
    """
    Apply calibration matrix to correct imbalance
    IQ_imbalance_matrix from factory calibration or measurement
    """
    iq_vector = torch.stack([I, Q], dim=-1)  # (..., 2)
    # Apply correction matrix
    iq_corrected = torch.matmul(iq_vector, IQ_imbalance_matrix.T)
    return iq_corrected[..., 0], iq_corrected[..., 1]
```

## Frequency Offset Estimation for Active Cancellation

When transmitting **cancellation waveform**, the TX leakage frequency shifts due to oscillator offset:

```python
def estimate_tx_leakage_frequency(rx_signal, tx_frequency, sample_rate):
    """
    Estimate leakage frequency for active cancellation synthesis
    """
    # Power spectral density
    psd = torch.abs(torch.fft.fft(rx_signal)) ** 2

    # Find peak near TX frequency
    freq_bins = torch.fft.fftfreq(len(rx_signal), 1/sample_rate)

    tx_bin = torch.argmin(torch.abs(freq_bins - tx_frequency))

    # Estimate peak within ±10 frequency bins
    search_range = slice(max(0, tx_bin-10), min(len(psd), tx_bin+10))
    peak_idx = torch.argmax(psd[search_range]) + max(0, tx_bin-10)

    estimated_freq = freq_bins[peak_idx]

    return estimated_freq
```

This frequency estimate drives the **phase-conjugate synthesis**:

```python
def generate_cancellation_waveform(leakage_freq, amplitude, duration, sample_rate):
    """Generate anti-phase signal to cancel TX leakage"""
    t = torch.arange(int(duration * sample_rate)) / sample_rate

    # Opposite phase to leakage
    cancellation = amplitude * torch.exp(-1j * 2 * np.pi * leakage_freq * t)

    return cancellation
```
