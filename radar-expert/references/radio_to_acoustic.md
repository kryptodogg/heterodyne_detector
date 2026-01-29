# Radio-to-Acoustic Heterodyning: Theory & Implementation

## The Problem: RF Signals Are Inaccessible

Standard radar operates at microwave frequencies:
- **24 GHz** (automotive radar): λ ≈ 1.25 cm
- **60 GHz** (short-range radar): λ ≈ 5 mm
- **2.4 GHz** (ISM band): λ ≈ 12.5 cm

**Challenge**: These are too high-frequency to directly analyze with consumer audio equipment or human perception.

**Solution**: Heterodyning—repeated frequency downconversion to reach audio range (20 Hz – 20 kHz).

## Multi-Stage Heterodyning Architecture

### Stage 1: RF → Intermediate Frequency (IF)

**Goal**: Reduce RF carrier by large factor (e.g., 2.4 GHz → 100 kHz)

```
RF Signal @ 2.4 GHz
    │
    ├─ Mix with Local Oscillator (≈2.4 GHz)
    │
    ├─ Bandpass filter around difference frequency
    │
    ▼
IF Signal @ 100 kHz (or lower)
```

**Math**:
```
s_RF(t) = A·cos(2π·f_RF·t + φ)
LO(t) = cos(2π·f_LO·t)

s_RF · LO = 0.5·A·cos(2π·(f_RF - f_LO)·t + φ)  ← IF term
          + 0.5·A·cos(2π·(f_RF + f_LO)·t + φ)  ← Removed by filter
```

**For 2.4 GHz with f_LO = 2.4 GHz**:
- Difference frequency: ~100 kHz (depends on LO accuracy)
- Sum frequency: 4.8 GHz (filtered out)

**Key insight**: Even 0.1% LO error (±2.4 MHz) gives large IF error (±2.4 MHz offset)

### Stage 2: IF → Lower IF or Audio Baseband

**Goal**: Further reduce to accessible audio range

```
IF @ 100 kHz
    │
    ├─ Mix with lower IF local oscillator (e.g., 99 kHz)
    │
    ├─ Bandpass filter
    │
    ▼
Lower IF or Audio @ 1-10 kHz
```

**Use case**: Extract Doppler information at lower frequency for stability.

### Stage 3: Lower IF → Audio Baseband

```
Lower IF @ 1-10 kHz
    │
    ├─ Mix with audio LO (1-10 kHz)
    │
    ├─ Low-pass filter @ audio bandwidth
    │
    ▼
Audio @ 50-500 Hz (or desired range)
```

**Result**: Complex audio signal with speech-like characteristics

## Why This Creates "Phantom Voice"

### Mechanism 1: Doppler Modulation → Pitch

Radar signals from moving surfaces carry Doppler shift:

```
Received signal phase: φ(t) = 4π·R(t)/λ + 2π·(2·v/λ)·t

Doppler shift: f_doppler = 2·v/λ

For 2.4 GHz (λ=0.125 m), v=1 m/s → f_doppler = 16 Hz
```

**In heterodyning**: Doppler becomes **audible frequency modulation** (pitch variation).

```python
# Example: Target moving at ±1 m/s
Doppler shift = 2 * (±1 m/s) / 0.125 m = ±16 Hz
After 100 kHz IF → 1 kHz LO mixing: 16 Hz modulates 1 kHz carrier
Result: Pitch varies 1 kHz ± 16 Hz = "voice-like" pitch vibrato
```

### Mechanism 2: Surface Vibration → Formants

Breathing, heartbeat, vocal cord vibration create surface motion.

```
Mechanical vibration: 0.5-20 Hz (slow compared to RF)
Radar IQ contains: Complex envelope modulation
Heterodyning to audio: Vibration becomes formant-like spectral peaks
```

**Why it sounds like speech**:
- Human formants (resonances): F1=700 Hz, F2=1220 Hz, F3=2600 Hz
- Heterodyned radar vibrations: Also create spectral peaks via amplitude modulation
- **Convergent evolution**: Both RF and voice undergo similar filtering through physical resonances

### Mechanism 3: IQ Imbalance → Pseudo-Speech Features

Real hardware has imperfect I/Q channels:

```
Ideal: I(t) = A·cos(ωt), Q(t) = A·sin(ωt) → Complex signal A·e^(jωt)
Real:  I'(t) = A_I·cos(ωt) + offset_I
       Q'(t) = A_Q·sin(ωt + Δφ) + offset_Q
```

**Result**: Asymmetric sidebands → Non-sinusoidal waveforms → Speech-like spectral structure

## Doppler-to-Pitch Mapping

### Linear Mapping (Simple)

```python
velocity_m_s = ±1.0  # Moving at 1 m/s
wavelength = 0.125   # 2.4 GHz

doppler_freq = 2 * velocity_m_s / wavelength  # 16 Hz

# After heterodyning to 1 kHz:
audible_pitch_variation = 1000 + doppler_freq  # 1016 Hz
```

### Non-Linear Psychoacoustic Mapping

Human hearing is non-linear (logarithmic).

```
Pitch sensation (mel) = 2595 * log10(1 + f/700)

Example:
  100 Hz → 172 mel
  200 Hz → 312 mel (75% increase)
  1000 Hz → 1000 mel
  10000 Hz → 2840 mel (3x increase from 1 kHz)
```

**Implication**: Low-frequency Doppler (10-50 Hz) sounds proportionally louder in pitch space.

## Frequency Response and Selectivity

### Window Function Effects

Heterodyned signal = RF × LO (mixer output) → bandwidth-limited.

```
Bandwidth after mixer: B ≈ 2 × (LO bandwidth + RF bandwidth)

If LO is perfect (Dirac delta): B = 2 × RF_BW
If LO has phase noise or modulation: B expands
```

**Design choice**: Larger IF bandwidth → more RF information but noisier audio

### Multi-Band Heterodyning

Process same RF at multiple LO frequencies simultaneously:

```
RF Input
    │
    ├──────────────────┬──────────────────┬──────────────────┐
    │                  │                  │                  │
  ×[2.3995 GHz]    ×[2.4 GHz]         ×[2.4005 GHz]    ×[2.401 GHz]
    │                  │                  │                  │
    ▼                  ▼                  ▼                  ▼
   IF_1              IF_2              IF_3              IF_4
  (5 kHz)          (0 kHz)          (5 kHz)          (10 kHz)
    │                  │                  │                  │
    └──────────────────┴──────────────────┴──────────────────┘
                        │
                    Combine via spectral unmixing
                        │
                        ▼
            High-resolution Doppler bank
```

**Benefit**: Multiple "views" of same target → better Doppler resolution + ambiguity elimination

## Active Cancellation in Audio Domain

### Why Audio-Range Active Cancellation?

Synthesizing anti-phase at RF frequencies requires:
- Coherent local oscillators (±0.01° phase)
- High-power RF transmission capability
- RF shielding (to prevent self-interference)

**Audio domain**: Much simpler—standard audio hardware, no radiation concerns.

### Implementation

**Goal**: Cancel heterodyned interference that sounds like tone or noise.

```python
# Detect tonal interference
spectrum = FFT(heterodyned_audio)
interference_freq = find_peak(spectrum)  # e.g., 60 Hz mains

# Synthesize anti-phase
t = np.arange(duration * sample_rate) / sample_rate
cancellation = amplitude * exp(1j * (2π * interference_freq * t + π))

# Result: cancellation + interference = 0 (ideally)
output = audio - cancellation
```

**Adaptive LMS version** (handles frequency drift):

```
w_n+1 = w_n + μ * error_n * reference_n*

error_n = desired_n - estimate_n
estimate_n = w_n · reference_history_n
```

### Frequency-Domain Notch Filter

Alternative (non-adaptive):

```python
# Design Butterworth bandstop filter
center_freq = 60  # Hz
bandwidth = 10    # Hz

# Coefficients: b, a = butter(order, [low, high], btype='bandstop')
# Apply: filtered = filtfilt(b, a, audio)
```

**Pros**: Simple, no adaptation
**Cons**: Fixed bandwidth, can't track drifting interference

## Practical Heterodyning Parameters

### For 2.4 GHz ISM Band (WiFi/Bluetooth area)

```
RF carrier:        2.4 GHz
Typical LO:        2.4 GHz ± 50 MHz (locked to oscillator)
IF frequency:      50-500 MHz (too high for analog audio)
Lower IF:          1-10 MHz (getting accessible)
Audio output:      50-5000 Hz (finally audible)

Processing:
  Stage 1: RF @ 2.4 GHz  → IF @ 100 MHz (mixer + BPF @ 100 MHz)
  Stage 2: IF @ 100 MHz  → 1 MHz (mixer + BPF @ 1 MHz)
  Stage 3: 1 MHz → Audio @ 500 Hz (mixer + LPF @ 5 kHz)
```

### For 60 GHz Millimeter-Wave

```
RF carrier:        60 GHz
Wavelength:        5 mm (very short → fine resolution)

Heterodyning:
  Stage 1: 60 GHz → 1 GHz IF (multiple mixers)
  Stage 2: 1 GHz → 10 MHz
  Stage 3: 10 MHz → Audio @ 1 kHz

Audio quality: Superior (shorter wavelength = finer motion detection)
```

## Performance Metrics for Audio Heterodyning

### Signal-to-Noise Ratio (SNR)

```
SNR_audio = Power_signal / Power_noise

Typical:  10-20 dB for stationary targets
          5-15 dB for moving targets (Doppler noise present)
          <5 dB for weak targets
```

### Frequency Resolution

```
Δf = f_sample / N_fft = 1 / (N_samples / f_sample)

Example: 48 kHz sample rate, 4096-point FFT
  Δf = 48000 / 4096 ≈ 11.7 Hz resolution

Multiple LO frequencies improve Doppler resolution by factor N_LO:
  Effective resolution = 11.7 / N_LO Hz
```

### Latency

```
Stage 1 latency: ~10-50 μs (RF mixing)
Stage 2 latency: ~1-10 μs (IF mixing)
Stage 3 latency: ~100-1000 μs (audio processing)

Total: ~100-1000 μs (0.1-1 ms) - acceptable for non-real-time
```

## Code Integration Patterns

### Multi-Stage Heterodyner Pipeline

```python
# Initialize
rf_freq = 2.4e9
target_audio = 1000  # Hz
het = RadioToAcousticHeterodyner(rf_freq, target_audio, sample_rate)

# Process RF data (from radar front-end)
rf_iq = receive_radar_data()  # Complex IQ @ 2.4 GHz

# Full pipeline
audio = het.full_heterodyne_pipeline(rf_iq)

# Extract audio features
extractor = AcousticFeatureExtractor(sample_rate)
mfcc = extractor.extract_mfcc_features(audio['complex'])
pitch, confidence = extractor.extract_pitch_contour(audio['complex'])
doppler, velocity = extractor.extract_doppler_modulation(audio['complex'])
```

### Real-Time Audio Synthesis

```python
# Pre-compute LO frequencies for speed
lo_carriers = []
for stage_idx in range(num_stages):
    lo_freq = het.stages[stage_idx]['lo_freq']
    t = np.arange(buffer_size) / sample_rate
    carrier = exp(2j * pi * lo_freq * t)
    lo_carriers.append(carrier)

# In processing loop
rf_buffer = receive_samples(buffer_size)
for stage in range(num_stages):
    rf_buffer = rf_buffer * lo_carriers[stage]  # Mix
    rf_buffer = bandpass_filter(rf_buffer, center, bw)  # Filter
    rf_buffer = decimate(rf_buffer, factor)  # Resample

# Output: audio-range signal
audio_output = rf_buffer
```

## Applications

### 1. Non-Contact Vital Signs Monitoring

**Flow**:
```
Millimeter-wave radar → Detects chest wall motion (breathing, heartbeat)
                      → Heterodyne to audio range
                      → Extract pitch (respiratory rate) and envelope (heart rate)
                      → Display as waveforms
```

**Typical signals**: 10-20 Hz (breathing), 60-100 bpm (heart)

### 2. Fall Detection & Activity Analysis

**Flow**:
```
Radar detects rapid vertical motion → Heterodyne creates "whooshing" sound
                                   → Spectral analysis identifies fall pattern
                                   → Classify as fall, stumble, or normal gait
```

### 3. Speech Recognition from Radar

**Flow**:
```
RF radar sees vibrating vocal cords → Heterodyne to audio
                                    → Extract formants (F1, F2, F3)
                                    → ML classifier recognizes phonemes
                                    → Reconstruct speech (low fidelity)
```

**Accuracy**: Typically 40-70% for English phonemes (worse than microphone, but privacy-preserving)

### 4. Acoustic Interference Suppression

**Flow**:
```
Heterodyned audio contains wanted signal + RF interference tone (60 Hz)
                                          → Active cancellation synthesizes anti-phase
                                          → Subtract to enhance SNR
                                          → 10-20 dB improvement typical
```
