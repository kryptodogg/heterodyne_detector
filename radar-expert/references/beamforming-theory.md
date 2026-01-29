# Beamforming Theory: From Geometry to Null Steering

## Physical Setup: Uniform Linear Array (ULA)

**Two-element receiver array** (most common for compact radars):

```
TX                     TX
o-------5cm-------o    (2 transmitters)
        |
       /|\
      / | \ targets at different angles θ
     /  |  \
    RX1 |   RX2  (2 receivers, 5 cm apart)
    o---|---o
       5cm
        (baseline)

Wavelength at 2.4 GHz: λ ≈ 12.5 cm
Baseline as fraction of wavelength: 5 cm / 12.5 cm = 0.4λ
```

## Steering Vector: The Fundamental Building Block

**Definition**: Complex vector encoding how a plane wave from angle θ arrives at each antenna.

For a signal arriving at angle θ (measured from broadside):

```
Signal at RX1: s(t)
Signal at RX2: s(t - τ)  where τ = time delay

The time delay corresponds to a phase shift:
Δφ = 2π · f · τ = 2π · f · (baseline/c) · sin(θ)

Let k = 2π/λ (wavenumber)
Then: Δφ = k · baseline · sin(θ)

Steering vector: a(θ) = [1, e^(j·k·d·sin(θ))]
                        └─ RX1  └─ RX2 with phase lag
```

### PyTorch Implementation

```python
def compute_steering_vector(baseline, wavelength, angle_deg):
    """
    Compute steering vector for given angle and geometry

    Returns: tensor of shape (N_angles, 2) for 2-element array
    """
    angles_rad = torch.tensor(angle_deg) * np.pi / 180.0
    k = 2 * np.pi / wavelength  # Wavenumber

    # Phase shift from baseline delay
    phase = k * baseline * torch.sin(angles_rad)

    # Steering vector: [1, e^(j·phase)]
    a = torch.stack([
        torch.ones_like(phase),
        torch.exp(1j * phase)
    ], dim=-1)

    return a  # Shape: (..., 2)
```

## Beamforming: Combining Antenna Outputs

**Goal**: Weight and combine RX1 + RX2 outputs to:
1. **Maximize** signal from a desired direction θ_target
2. **Minimize** noise from other directions

```
y = w_1 * RX1 + w_2 * RX2

where w = [w_1, w_2] are complex weights (amplitude + phase)
```

### Conventional Beamformer (Delay-and-Sum)

**Simplest approach**: Weight each antenna to align phases at target direction.

```
w_conventional(θ_target) = a(θ_target)  [Normalized]

Result: Signal from θ_target constructively interferes
        Signal from other θ has random phase → noise-like
```

**Problem**: Broad beam pattern, poor interference suppression.

```python
def delay_and_sum(rx1, rx2, baseline, wavelength, target_angle_deg):
    """Conventional beamformer"""
    a = compute_steering_vector(baseline, wavelength, target_angle_deg)
    weights = a / torch.linalg.norm(a)  # Normalize

    output = torch.dot(weights.conj(), torch.stack([rx1, rx2], dim=0))
    return output
```

## MVDR Beamformer: Maximum Variance Distortionless Response

**Goal**: Pass target signal undistorted while minimizing output power (noise + interference).

### Mathematical Derivation

**Optimization problem**:
```
Minimize: E[|y|²] = E[|w^H x|²] = w^H R w

Subject to: w^H a(θ_target) = 1  [Unit gain at target]

Where R = E[x x^H] is the covariance matrix of received signals
      a(θ) is steering vector
      x = [RX1, RX2]^T
```

**Lagrange multiplier solution**:
```
w_mvdr = R^(-1) a(θ_target) / (a^H(θ_target) R^(-1) a(θ_target))
         └─ Inverse of covariance   └─ Normalization
```

**Key insight**: MVDR achieves **null steering** (power notches) in interference directions.

### Step-by-Step Implementation

```python
def mvdr_beamformer(rx1, rx2, baseline, wavelength, target_angle_deg):
    """
    MVDR beamformer: Adaptive null steering

    Input:
        rx1, rx2: Received signals (complex)
        baseline: Distance between RX1 and RX2
        wavelength: Wavelength = c / f_center
        target_angle_deg: Desired beamsteering angle

    Output:
        Beamformed output with interference nulls
    """

    # 1. Stack signals
    x = torch.stack([rx1, rx2], dim=0)  # (2, N_samples)

    # 2. Compute sample covariance matrix
    # R = (1/N) * sum_n [x_n * x_n^H]
    R = torch.matmul(x, x.conj().T) / x.shape[1]

    # 3. Regularization (Diagonal Loading)
    # Prevents singular matrix and stabilizes against model mismatch
    trace = torch.trace(R)
    lambda_reg = 1e-3 * trace  # 0.1% of trace energy
    R_reg = R + lambda_reg * torch.eye(2, dtype=R.dtype, device=R.device)

    # 4. Inverse
    R_inv = torch.linalg.inv(R_reg)

    # 5. Steering vector at target angle
    a = compute_steering_vector(baseline, wavelength, target_angle_deg)
    a = a.squeeze()  # (2,) for single angle

    # 6. MVDR weights
    numerator = torch.matmul(R_inv, a)
    denominator = torch.matmul(a.conj(), torch.matmul(R_inv, a))
    w_mvdr = numerator / denominator

    # 7. Beamform
    y = torch.matmul(w_mvdr.conj(), x)

    return y, w_mvdr
```

### MVDR Beam Pattern

**Spatial response** (power as function of angle):

```python
def compute_beam_pattern(w_beamformer, baseline, wavelength, angle_sweep_deg):
    """
    Compute spatial response of beamformer

    Returns power response at each angle
    """
    angles_rad = torch.tensor(angle_sweep_deg) * np.pi / 180.0

    # Steering vectors at all angles
    angles_grid = angles_rad.view(-1, 1)  # (N_angles, 1)
    k = 2 * np.pi / wavelength
    phase_all = k * baseline * torch.sin(angles_grid)  # (N_angles, 1)

    # Steering matrix: each row is a(θ_i)
    a_all = torch.cat([
        torch.ones_like(phase_all),
        torch.exp(1j * phase_all)
    ], dim=1)  # (N_angles, 2)

    # Response: |w^H * a(θ)|^2
    response = torch.abs(torch.matmul(a_all, w_beamformer)) ** 2

    return response

# Plot pattern
angles = torch.linspace(-90, 90, 180)
pattern = compute_beam_pattern(w_mvdr, baseline=0.05, wavelength=0.125, angle_sweep_deg=angles)
```

**Expected pattern**:
```
Power (dB)

    |     /\              θ = target direction
    |    /  \
    |   /    \       /\
 20 |  /      \     /  \
    |_/_______\____/____\___  θ (degrees)
   -90 -45   0    45   90
         ↑   ↑      ↑
      Null Null  Null    (interference directions)

Beamwidth at -3dB: ~20-30° for 2-element array
Main lobe gain: ~6 dB (2x power from 2 antennas)
Null depth: 20-30 dB (depending on SNR and array size)
```

## Adaptive LMS: Tracking Time-Varying Interference

MVDR assumes **quasi-stationary** signals (statistics don't change much).

**Problem**: Real interference is dynamic. Need **adaptive** weights.

### LMS Algorithm (Least Mean Squares)

**Goal**: Iteratively adjust weights to minimize squared error.

```
y_desired = Clean signal (desired output)
y_estimated = w^H * x  (current estimate)

error = y_desired - y_estimated

Update: w[n+1] = w[n] + μ * error[n] * x[n]^*

where μ = learning rate (step size)
      ^* = complex conjugate
```

**Intuition**:
- If error is large, make bigger weight updates
- Updates move in direction of error gradient
- Converges to Wiener solution (optimal for Gaussian signals)

### PyTorch LMS Implementation

```python
def adaptive_lms_filter(target_rx, reference_rx, desired_output,
                       filter_length=128, learning_rate=0.01, max_iters=None):
    """
    Adaptive LMS cancellation filter

    Learns to remove noise from target_rx using reference_rx as model

    target_rx: Signal with noise (to be cleaned)
    reference_rx: Noise reference (correlated with noise in target)
    desired_output: Desired clean signal (if known; otherwise use target)
    """

    if max_iters is None:
        max_iters = len(target_rx)

    # Initialize weights
    w = torch.zeros(filter_length, dtype=target_rx.dtype, device=target_rx.device)

    # Output and error tracking
    y_out = torch.zeros(max_iters, dtype=target_rx.dtype, device=target_rx.device)
    error = torch.zeros(max_iters, dtype=target_rx.dtype, device=target_rx.device)

    for n in range(max_iters):
        # Reference signal window (last filter_length samples)
        start_idx = max(0, n - filter_length)
        x_n = reference_rx[start_idx:n+1]

        # Pad to filter_length if needed
        if len(x_n) < filter_length:
            x_n = torch.cat([torch.zeros(filter_length - len(x_n),
                                        dtype=reference_rx.dtype,
                                        device=reference_rx.device),
                            x_n])

        # Filter estimate
        y_n = torch.dot(w.conj(), x_n)
        y_out[n] = y_n

        # Error
        err = desired_output[n] - y_n
        error[n] = err

        # Weight update
        w = w + learning_rate * err * x_n.conj()

    # Cleaned output
    cleaned = target_rx[:max_iters] - y_out

    return cleaned, w, error
```

### LMS Convergence Analysis

**Learning rate μ** controls convergence:

```
Optimal μ: μ_opt = 1 / (trace(R) × N_taps)

For stability: 0 < μ < 2 / λ_max(R)

Example:
  R eigenvalues: [1.0, 0.1]
  λ_max = 1.0
  Max stable μ = 2 / 1.0 = 2.0
  Conservative choice: μ = 0.1 × 2.0 = 0.2

Too small (μ = 0.01): Slow learning, good stability
Too large (μ = 1.0): Fast learning, risk of divergence
```

## Combined MVDR + LMS Strategy

**Why combine?**
- MVDR: Excellent for **static** known-geometry interference
- LMS: Excellent for **dynamic, unknown** interference patterns

**Strategy**:
```
1. MVDR initialization: Compute w_mvdr from first snapshot
2. LMS refinement: Use w_mvdr as seed, adapt with LMS
3. Result: Fast initial suppression + continuous adaptation
```

```python
def mvdr_plus_lms(rx1, rx2, baseline, wavelength, target_angle_deg,
                  filter_length=128, learning_rate=0.01):
    """
    Two-stage beamformer: MVDR initialization + LMS adaptation
    """

    # Stage 1: MVDR (first 1000 samples for covariance estimation)
    x_init = torch.stack([rx1[:1000], rx2[:1000]], dim=0)
    R = torch.matmul(x_init, x_init.conj().T) / 1000
    R_reg = R + 1e-3 * torch.trace(R) * torch.eye(2)
    R_inv = torch.linalg.inv(R_reg)

    a = compute_steering_vector(baseline, wavelength, target_angle_deg).squeeze()
    w_init = torch.matmul(R_inv, a) / torch.dot(a.conj(), torch.matmul(R_inv, a))

    # Stage 2: LMS starting from w_init
    x_stacked = torch.stack([rx1, rx2], dim=0)
    w = w_init.clone()

    y_out = torch.zeros(len(rx1), dtype=rx1.dtype, device=rx1.device)

    for n in range(len(rx1)):
        # Form reference signal (use both RX channels as reference)
        x_n = x_stacked[:, max(0, n-filter_length//2):n+1]

        # Project onto filter_length dimension
        if x_n.shape[1] < filter_length // 2:
            x_n_padded = torch.cat([
                torch.zeros((2, filter_length // 2 - x_n.shape[1]),
                           dtype=rx1.dtype, device=rx1.device),
                x_n
            ], dim=1)
        else:
            x_n_padded = x_n[:, -filter_length//2:]

        # Flatten: (2 × filter_length/2,)
        x_flat = x_n_padded.flatten()

        # Filter
        y_n = torch.dot(w[:len(x_flat)].conj(), x_flat)
        y_out[n] = y_n

        # Error (we want to suppress leakage, so desire zero output)
        err = 0.0 - y_n

        # Adapt
        w[:len(x_flat)] = w[:len(x_flat)] + learning_rate * err * x_flat.conj()

    return y_out
```

## Array Geometry Impact on Resolution

### DOA (Direction of Arrival) Resolution

**Angular resolution vs. baseline**:

```
For ULA with 2 elements:

Δθ ≈ λ / (2 · baseline)  [in radians]
   ≈ 57.3 · λ / baseline  [in degrees]

Example at 2.4 GHz (λ ≈ 0.125 m):
  baseline = 5 cm:  Δθ ≈ 1.4°    (good resolution)
  baseline = 2 cm:  Δθ ≈ 3.6°    (poor resolution)
  baseline = 10 cm: Δθ ≈ 0.7°    (very good)
```

### Ambiguity Wrapping

**Problem**: Phase only known mod 2π.

```
For sine argument: sin(θ) only defined up to ±π/2

Wrapped DOA range: [−λ/2, +λ/2] in space

Beyond this, angles "wrap" (ambiguous)
```

**Solution**: Use **phase unwrapping** or larger arrays (UPA/UBA).

## Real-World Considerations

### Mutual Coupling

Antennas interact electromagnetically. Changes effective spacing.

**Effect**: Steering vector accuracy ↓, null depth ↓

**Mitigation**:
- Physically separate antennas (λ/2 spacing at minimum)
- Apply coupling matrix correction (from calibration)

### Model Mismatch

Assumed ULA geometry doesn't match reality.

**Effect**: MVDR nulls don't align with actual interference

**Mitigation**:
- Verify geometry with calibration signal
- Reduce MVDR null depth (higher regularization λ)
- Rely more on LMS adaptation

### Finite Sample Effects

Covariance matrix estimated from limited snapshots.

**Required samples**: N_samples >> N_channels² (for 2 RX, need ~4 samples minimum)

**Practical rule**: N_samples = 100-1000 × N_channels for stable MVDR

## Code Patterns for Integration

### Pattern 1: Scan DOA Space

```python
def scan_beamformer_response(rx1, rx2, baseline, wavelength, angle_step=1.0):
    """
    Sweep beamformer across all angles to find targets

    Returns: (angles, power_response)
    """
    angles = torch.arange(-90, 90, angle_step)
    responses = []

    for angle in angles:
        y, w = mvdr_beamformer(rx1, rx2, baseline, wavelength, float(angle))
        power = torch.mean(torch.abs(y) ** 2)
        responses.append(power)

    return angles, torch.stack(responses)
```

### Pattern 2: Adaptive Null Steering

```python
def adaptive_null_steering(rx1, rx2, target_angle, interference_angles,
                          baseline, wavelength):
    """
    MVDR with constraints to maintain nulls at known interference
    """
    # Build constraint matrix (pass target, null at interference)
    # Advanced: LCMV (Linearly Constrained Minimum Variance)
    # ...implementation details...
    pass
```
