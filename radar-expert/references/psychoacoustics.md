# Psychoacoustics of Heterodyned Radar Signals

## Why Heterodyned Radar Sounds Like Speech

The human auditory system evolved to process sound, not electromagnetic signals. Yet heterodyned radar produces signals that **sound strikingly speech-like**. This is not coincidence—it's convergent psychoacoustics.

### Three Perceptual Mechanisms

#### 1. Formant Recognition

**Definition**: Formants are resonant frequencies of the vocal tract.

```
Human speech formants (typical male):
  F1 (jaw opening): 600-900 Hz
  F2 (tongue position): 800-2000 Hz
  F3 (pharynx): 2000-3500 Hz

These are NOT generated directly by vocal cords (which vibrate at 50-250 Hz).
Instead, they emerge from **filtering** the glottal excitation through vocal tract resonances.
```

**Why heterodyned radar has formants**:

```
Radar sees vibrating surface → Complex amplitude modulation
Heterodyning process = nonlinear filtering
Result: Spectral peaks at resonant frequencies (coincidentally similar to formants)

Example: Breathing at 0.25 Hz
  RF phase modulation: 4π·R·cos(2π·0.25·t)
  After heterodyne to 1 kHz: Appears as 1 kHz ± modulation sidebands
  Sideband spacing reflects surface vibration modes (lung/chest cavity resonances)
  → Sounds like formant-like peaks
```

#### 2. Pitch Perception and Weber's Law

**Weber's Law**: Perception of change is proportional to baseline intensity.

```
Δf / f = constant (approximately)

Example:
  100 Hz → 101 Hz is noticeable (1% change)
  1000 Hz → 1010 Hz is same perceptual distance
  10000 Hz → 10100 Hz would need 1% change
```

**Implication for heterodyned radar**:

```
Doppler shift at IF frequency appears as LARGE pitch change perceptually:

2.4 GHz RF, 1 m/s motion:
  Doppler = 2 * 1 / 0.125 = 16 Hz

After heterodyne to 1 kHz base:
  1000 Hz + 16 Hz modulation = 1.6% pitch variation
  Perceptually: Large pitch slide (like human prosody/intonation)
```

**Speech prosody typically**: 5-10% pitch variation (100 Hz in 1 kHz base)

Heterodyne naturally creates similar prosody!

#### 3. Periodicity and Stationarity

**Haas Effect**: Humans perceive repetitive signals as continuous (speech-like).

```
Single chirp: Isolated sound
100 chirps/sec: Perceived as continuous tone or vowel
Modulated at 0.5-20 Hz: Speech-like formant evolution
```

**Heterodyned radar**: Generates **pseudo-periodic** output from continuous RF.

```
Continuous RF from vibrating surface:
  Deterministic amplitude modulation (vibration envelope)
  Phase modulation (Doppler velocity)
  = Combines into pseudo-periodic complex signal

Human auditory system: Interprets periodicity as *phonemes*
```

## Mel-Frequency Cepstral Coefficients (MFCCs)

### Why MFCCs for Radar?

MFCCs compress speech characteristics into 13-39 coefficients:

```
MFCC extraction:
  1. Compute spectrogram (FFT over time)
  2. Apply mel-scale warping (logarithmic frequency, matching hearing)
  3. Apply mel-scaled triangular filterbank (20-128 filters)
  4. Compute power in each band
  5. Log-scale the power
  6. Apply DCT (Discrete Cosine Transform)
  7. Keep first 13-39 coefficients
```

### MFCC Robustness

MFCCs are robust to:
- ✅ Noise (averaging over frequency bands)
- ✅ Time-varying signals (delta/delta-delta add velocity + acceleration)
- ✅ Loudness changes (log scaling)
- ❌ Phase distortions (sensitive to phase coherence)

**For heterodyned radar**: MFCCs work well because:

1. **Amplitude modulation is preserved**: Breathing/vibration envelope encoded
2. **Frequency structure captured**: Formant-like peaks detected
3. **Time dynamics included**: Pitch slides and prosody matter

### MFCC Application to Radar

```python
# MFCCs for radar activity classification
from acoustic_features import AcousticFeatureExtractor

extractor = AcousticFeatureExtractor(sample_rate=48000)
mfcc = extractor.extract_mfcc_features(heterodyned_audio, n_mfcc=13)

# Classification
# mfcc[0:3]: Energy-related coefficients (activity level)
# mfcc[3:8]: Mid-frequency structure (breathing patterns)
# mfcc[8:13]: High-frequency structure (detail, speech-likeness)

if mfcc[0] > threshold_activity:
    if mfcc[3:8].std() > threshold_motion:
        classify_as('walking')  # Rhythmic breathing patterns
    else:
        classify_as('stationary')  # Steady breathing
```

## Auditory Scene Analysis (ASA)

### Gestalt Principles in Hearing

The brain organizes sounds via:

1. **Spectral Grouping**: Frequencies that form harmonic series are grouped
   ```
   Harmonics of 100 Hz: 100, 200, 300, 400, 500 Hz → Perceived as single voice
   Inharmonic: 100, 220, 350, 500 Hz → Perceived as multiple sources
   ```

2. **Temporal Coherence**: Events starting/stopping together are grouped
   ```
   Same onset time, correlated amplitude envelope → Same source
   Different onsets → Different sources
   ```

3. **Continuity Illusion**: Sounds are filled in if masked by louder sounds
   ```
   Tone A at 1000 Hz
   Tone B at 5000 Hz (overlapping in time, louder)
   → We still "hear" Tone A (filled in by brain)
   ```

### Implication for Heterodyned Radar

Heterodyned radar naturally satisfies ASA principles:

```
Multi-band heterodyning (multiple LO frequencies):
  Band 1 @ 500 Hz: Breathing (~0.3 Hz modulation)
  Band 2 @ 1 kHz: Movement (~1-5 Hz modulation)
  Band 3 @ 2 kHz: Cardiac (~1-2 Hz modulation)

Auditory grouping:
  Similar modulation rates → Perceived as same "voice" (person)
  Separate modulation → Perceived as different people
```

## Critical Bands and Masking

### Critical Band Concept

Frequency resolution of human hearing is limited (not uniform).

```
Critical bandwidth ≈ 25 * (f_Hz / 1000)^0.69 Hz

Examples:
  100 Hz: CBW ≈ 50 Hz
  1 kHz: CBW ≈ 100 Hz
  10 kHz: CBW ≈ 400 Hz
```

**Implication**: Sounds within a critical band are harder to distinguish.

### Simultaneous Masking

A loud sound masks quieter sounds within critical band:

```
Loud signal @ 1000 Hz masks:
  Quiet signal @ 900-1100 Hz (within critical band)
  Not @ 1500 Hz (outside critical band)

Masking depth: Can be 20-40 dB
```

### Temporal Masking

Loud event masks quieter sounds before/after:

```
Pre-masking: ~5-20 ms (quiet sounds before loud event masked)
Post-masking: ~100-300 ms (quiet sounds after loud event masked)
```

**For active audio cancellation**:
```
Cancellation signal timing critical:
  Too early: Pre-masking hides cancellation signal
  Too late: Post-masking allows original to be heard
  Optimal: Anti-phase signal aligned to sample-level accuracy (microseconds)
```

## Formant-Based Classification

### Formant Extraction from Heterodyned Audio

```python
# Extract formants via LPC (Linear Predictive Coding)
from acoustic_features import AcousticFeatureExtractor

formants, powers = extractor.extract_formant_frequencies(audio)

# Compare to known patterns
human_formants = {
    'male': [700, 1220, 2600],      # Hz
    'female': [850, 1430, 2800],
    'child': [1000, 1600, 3000],
}

# Classification
if formants[0] > 800 and formants[1] > 1300:
    classify_as('female_like')
else:
    classify_as('male_like')
```

### Why Formant-Based Classification Works

```
Formants determined by:
  - Resonant cavities (vocal tract length)
  - Articulation (tongue position, jaw opening)
  - For radar: Surface material, shape, vibration patterns

Heterodyned radar formants reflect:
  - Chest/lung resonances (breathing): F1-like ~100-500 Hz
  - Heart/rib cage modes: F2-like ~500-2000 Hz
  - Arterial pulsation: F3-like ~2000+ Hz

→ Can distinguish human from animal, health status, emotion
```

## Pitch Contour and Intonation

### Fundamental Frequency Tracking

```python
pitch_contour, confidence = extractor.extract_pitch_contour(audio)

# Pitch contour characteristics
mean_pitch = np.mean(pitch_contour[confidence > 0.3])
pitch_range = np.max(pitch_contour) - np.min(pitch_contour)
pitch_acceleration = np.diff(np.diff(pitch_contour))
```

### Linguistic vs. Emotional Prosody

**Statement**: Pitch falls at end
```
"The meeting is at 3 PM" ↘ (falling)
```

**Question**: Pitch rises at end
```
"The meeting is at 3 PM?" ↗ (rising)
```

**Emotion** (anger, joy, sadness):
```
Angry: High pitch, rapid acceleration, wide range
Sad: Low pitch, slow acceleration, narrow range
Neutral: Mid pitch, moderate range
```

**For heterodyned radar**:
```
Doppler velocity profile creates pitch contour
Velocity acceleration = pitch acceleration
→ Emotional state / activity intensity detectable from Doppler dynamics
```

## Phantom Audio Synthesis

### Auditory Streaming

When sounds move in frequency, we perceive them as continuous streams.

```
Stimulus: 400 Hz → 500 Hz → 600 Hz (100 ms each)
Perception: Single gliding tone (smooth pitch slide)

If separated by gaps: Percieved as separate tones
```

### Pitch Shift Illusion

Shepard tone: Pitch appears to ascend continuously but returns to start.

```
Construct: Superposition of sinusoids at octaves
  500 Hz, 1000 Hz, 2000 Hz, 4000 Hz (each rising in frequency)
  Amplitude envelope: Gaussian peak @ 1 kHz

Perception: Continuous ascending pitch, but never gets higher!
```

**Application to heterodyned radar**:
```
Multi-band heterodyning creates illusion of unified "voice"
Each band contributes to pitch and timbre
→ Synthesized "phantom voice" sounds more natural than any single band alone
```

### Formant Trajectory and Speech Synthesis

**Concatenative synthesis**: String together recorded phonemes (meh, duh, buh)

**Formant synthesis**: Interpolate formant trajectories over time

**For heterodyned radar**:
```python
# Extract formant trajectories
formants_over_time = []
for frame in audio_frames:
    f = extract_formants(frame)
    formants_over_time.append(f)

# Synthesize "phantom voice" by enhancing formant trajectories
synthetic_audio = synthesize_formant_trajectory(formants_over_time)

# Result: Heterodyned radar + formant enhancement = very speech-like output
```

## Hearing Loss and Age

### Age-Related Hearing Loss (Presbycusis)

Hearing sensitivity decreases with age:

```
Typical audiogram:
  Age 20: Flat to 15 kHz
  Age 40: Gradual drop above 3 kHz
  Age 60: Significant loss above 2 kHz, some below 1 kHz
  Age 80: Severe loss above 1 kHz

Impact on heterodyned radar perception:
  Higher frequency bands (>2 kHz) become inaudible
  Pitch perception shifts (tilted towards lower frequencies)
```

### Hearing Aid Processing

Modern hearing aids use:
- Adaptive multichannel compression
- Noise suppression algorithms
- Frequency transposition (shift high frequencies lower for accessible range)

**Implication**: Using hearing aid signal processing on heterodyned radar could:
- Enhance weak signals
- Suppress RF interference
- Make high-frequency components audible

## Practical Psychoacoustic Tuning

### Perceived Loudness vs. Physical Intensity

Loudness perception: Logarithmic + frequency-dependent

```python
# A-weighting (matches human hearing sensitivity)
A_weight = [
    (100, -26.2), (500, -3.2), (1000, 0), (5000, 4), (10000, 12)
]

# Apply A-weighting to heterodyned spectrum
spectrum_aweighted = spectrum * A_weight_curve
perceived_loudness = compute_loudness(spectrum_aweighted)
```

### Optimal Audio Presentation Frequency

For heterodyned radar, optimal presentation depends on:

```
1. Activity type:
   Breathing: 0.2-1 Hz (too low for audio, heterodyne to 500-1000 Hz)
   Heart rate: 60-100 bpm = 1-2 Hz (heterodyne to 1-2 kHz for perceptual prominence)
   Motion: 1-10 Hz (heterodyne to 2-5 kHz for visibility)

2. Discrimination task:
   Two targets (e.g., two people): Separate into different frequency bands
   Many targets (e.g., crowd): Use harmonically related bands (octaves)

3. Fatigue:
   Continuous sine tone: Becomes inaudible after ~30 seconds (adaptation)
   Solution: Time-varying heterodyne frequency or modulation
```

## Applications

### Hearing Impaired Heterodyned Radar Interface

```python
# For users with high-frequency hearing loss:
het = RadioToAcousticHeterodyner(rf_freq=2.4e9,
                                target_audio_freq=500,  # Lower frequency
                                sample_rate=sample_rate)

# Transpose high-frequency information downward
audio_low = het.full_heterodyne_pipeline(rf_signal)
audio_low_enhanced = enhance_formants(audio_low)  # Make structure audible

# Play through hearing aid
hearing_aid.process(audio_low_enhanced)
```

### Multimodal Display

```
Visual display (primary):
  - Spectrogram
  - Vital signs numbers
  - Activity classification

Audio display (secondary/confirmation):
  - Heterodyned signal at frequency optimized for discrimination
  - Pitch contour conveys velocity
  - Amplitude conveys signal strength
```

### Blind Navigation Using Heterodyned Radar

```python
# Radar scans room → Converts to 3D spatial audio (spatialization)
# Obstacle at 2 meters in front → Low pitch tone, centered
# Person 1 meter to left → Shifted pitch, left speaker
# Movement towards you → Rising pitch
# Movement away → Falling pitch
```
