#!/usr/bin/env python3
"""
Frequency Sweep Utility: Generate DOA response maps and validate beamforming

Creates angle-of-arrival heatmaps for diagnostics and calibration.
"""

import numpy as np
import torch


def create_steering_vectors(baseline, wavelength, angles_deg, n_channels=2):
    """
    Generate steering vectors for all angles

    Args:
        baseline: Antenna spacing in meters
        wavelength: RF wavelength in meters
        angles_deg: Numpy array of angles [0, 180] degrees
        n_channels: Number of receive channels

    Returns:
        Steering matrix (N_angles, n_channels) complex
    """

    angles_rad = np.radians(angles_deg)
    k = 2 * np.pi / wavelength

    # Phase shifts for each channel
    # Channel 0 at origin: phase = 0
    # Channel i at position i*d: phase = k*i*d*sin(angle)
    phase_matrix = np.zeros((len(angles_deg), n_channels))

    for ch in range(n_channels):
        phase_matrix[:, ch] = k * ch * baseline * np.sin(angles_rad)

    # Steering vectors: exp(j*phase)
    steering = np.exp(1j * phase_matrix)

    return torch.tensor(steering, dtype=torch.complex64)


def beamformer_response_map(rx_signals, baseline, wavelength,
                           angle_resolution=1.0, beamformer_type='conventional'):
    """
    Create 2D DOA response for visualization

    Args:
        rx_signals: List of (time_domain_signal, channel_index)
        baseline: Antenna baseline
        wavelength: Wavelength
        angle_resolution: Angle step in degrees
        beamformer_type: 'conventional', 'mvdr', or 'lcmv'

    Returns:
        (angles, responses) where responses[i] = power at angles[i]
    """

    angles = np.arange(0, 180, angle_resolution)
    responses = np.zeros(len(angles))

    for idx, angle in enumerate(angles):
        a = create_steering_vectors(baseline, wavelength, np.array([angle]), n_channels=2)
        a = a.squeeze()  # (2,)

        if beamformer_type == 'conventional':
            # Delay-and-sum
            w = a / np.linalg.norm(a)
        elif beamformer_type == 'mvdr':
            # MVDR (requires input signal)
            # Simplified: use magnitude of steering vector (no adaptation)
            w = a
        else:
            raise ValueError(f"Unknown beamformer: {beamformer_type}")

        # Combine channels
        if len(rx_signals) == 2:
            rx1, rx2 = rx_signals[0], rx_signals[1]
            output = w[0].conj() * rx1 + w[1].conj() * rx2
            power = torch.mean(torch.abs(output) ** 2)
            responses[idx] = float(power)

    return angles, responses


def mvdr_response_map(rx1, rx2, baseline, wavelength, angle_resolution=1.0):
    """
    MVDR spatial response (includes adaptation)

    Args:
        rx1, rx2: Received signals (complex tensors)
        baseline: Antenna baseline
        wavelength: Wavelength
        angle_resolution: Angle step

    Returns:
        (angles, mvdr_response_db)
    """

    angles = np.arange(0, 180, angle_resolution)
    responses = np.zeros(len(angles))

    # Covariance matrix from input
    x = torch.stack([rx1, rx2], dim=0)
    R = torch.matmul(x, x.conj().T) / x.shape[1]
    R_reg = R + 1e-3 * torch.trace(R) * torch.eye(2, dtype=R.dtype)
    R_inv = torch.linalg.inv(R_reg)

    for idx, angle in enumerate(angles):
        # Steering vector
        a = create_steering_vectors(baseline, wavelength, np.array([angle]), n_channels=2)
        a = a.squeeze()

        # MVDR formula: response = 1 / (a^H R^-1 a)
        numerator = torch.matmul(a.conj(), torch.matmul(R_inv, a))
        response = 1.0 / (torch.abs(numerator) + 1e-12)
        responses[idx] = float(response)

    # Convert to dB (normalize to max)
    responses_db = 10 * np.log10(responses / np.max(responses) + 1e-12)

    return angles, responses_db


def null_steering_angle(beamformer_weights, baseline, wavelength):
    """
    Find angles where beamformer produces nulls (minima)

    Args:
        beamformer_weights: Complex weight vector
        baseline: Antenna spacing
        wavelength: Wavelength

    Returns:
        List of null angles in degrees
    """

    angles, responses = beamformer_response_map(
        [torch.ones(1000), torch.ones(1000)],  # Placeholder signals
        baseline, wavelength, angle_resolution=0.5
    )

    # Find local minima
    diff = np.diff(responses)
    null_indices = np.where(np.diff(np.sign(diff)) > 0)[0]

    null_angles = angles[null_indices + 1]

    return null_angles


def frequency_dependent_resolution(center_freq, baseline, freq_range=None):
    """
    Compute DOA resolution vs. frequency

    Resolution improves at higher frequencies (shorter wavelength)

    Args:
        center_freq: Center frequency in Hz
        baseline: Antenna spacing in meters
        freq_range: Frequency range to sweep (default: ±50% around center_freq)

    Returns:
        (frequencies, resolutions_deg)
    """

    c = 299792458.0  # Speed of light

    if freq_range is None:
        freq_range = np.linspace(0.5 * center_freq, 1.5 * center_freq, 50)
    else:
        freq_range = np.linspace(freq_range[0], freq_range[1], 50)

    wavelengths = c / freq_range
    resolutions = 57.3 * wavelengths / baseline  # Convert rad to deg

    return freq_range / 1e9, resolutions  # Return GHz


def generate_test_signal(baseline, wavelength, angle_deg, duration_sec, sample_rate_hz):
    """
    Generate test signal from specific angle (for validation)

    Args:
        baseline: Antenna spacing
        wavelength: Wavelength
        angle_deg: Source angle
        duration_sec: Signal duration
        sample_rate_hz: ADC sample rate

    Returns:
        (rx1_signal, rx2_signal) simulating arrival from angle_deg
    """

    n_samples = int(duration_sec * sample_rate_hz)
    t = np.arange(n_samples) / sample_rate_hz

    # Plane wave from angle_deg
    k = 2 * np.pi / wavelength
    angle_rad = np.radians(angle_deg)

    # Phase difference due to baseline
    phase_diff = k * baseline * np.sin(angle_rad)

    # Complex amplitude (1 + 0j for unit amplitude)
    amplitude = 1.0 + 0.0j

    # RX1 (reference)
    rx1 = amplitude * np.exp(1j * 2 * np.pi * 100e3 * t)  # 100 kHz IF frequency

    # RX2 (delayed by phase_diff)
    rx2 = amplitude * np.exp(1j * (2 * np.pi * 100e3 * t + phase_diff))

    return torch.tensor(rx1, dtype=torch.complex64), \
           torch.tensor(rx2, dtype=torch.complex64)


def plot_beampattern(angles, responses, title="Beamformer Response"):
    """
    Print ASCII plot of beamformer response

    Args:
        angles: Angle array (degrees)
        responses: Response magnitude (linear or dB)
        title: Plot title
    """

    print(f"\n{title}")
    print("=" * 70)

    # Normalize to 0-20 range for display
    if np.min(responses) < 0:
        # dB scale
        responses_norm = responses - np.min(responses)
        responses_norm = 40 * responses_norm / np.max(responses_norm)
    else:
        # Linear scale
        responses_norm = 40 * responses / np.max(responses)

    # Create ASCII plot
    width = 70
    height = 20

    print(f"Angle (deg) →")
    print(f"↓ Power")

    for row in range(height, 0, -1):
        line = f"{20 - row:2d} |"
        for col in range(width):
            angle_idx = int(col * len(angles) / width)
            if responses_norm[angle_idx] >= 40 * row / height:
                line += "█"
            else:
                line += " "
        print(line)

    print("    +" + "─" * width)
    print("    " + " " * (width // 4) + "90°" + " " * (width // 2))

    print(f"\nResponse range: {np.min(responses):.2f} to {np.max(responses):.2f}")


if __name__ == "__main__":
    print("Frequency Sweep & DOA Analysis")
    print("=" * 70)

    # Example: 2.4 GHz radar with 5 cm baseline
    center_freq = 2.4e9
    baseline = 0.05  # 5 cm
    wavelength = 299792458 / center_freq

    print(f"Center Frequency: {center_freq/1e9:.2f} GHz")
    print(f"Wavelength: {wavelength*100:.2f} cm")
    print(f"Baseline: {baseline*100:.2f} cm")
    print(f"DOA Resolution: {57.3 * wavelength / baseline:.2f}°\n")

    # Frequency-dependent resolution
    freqs, resolutions = frequency_dependent_resolution(center_freq, baseline)
    print(f"Resolution at 2.0 GHz: {resolutions[0]:.2f}°")
    print(f"Resolution at 2.4 GHz: {resolutions[len(resolutions)//2]:.2f}°")
    print(f"Resolution at 2.8 GHz: {resolutions[-1]:.2f}°\n")

    # Test signal from 30° angle
    rx1, rx2 = generate_test_signal(baseline, wavelength, angle_deg=30,
                                   duration_sec=0.01, sample_rate_hz=10e6)

    # Compute beamformer response
    angles, responses = beamformer_response_map(
        [rx1, rx2], baseline, wavelength, angle_resolution=2.0
    )

    # Find peak
    peak_angle = angles[np.argmax(responses)]
    print(f"Beamformer peak at: {peak_angle:.1f}° (expected 30°)")

    plot_beampattern(angles, responses)
