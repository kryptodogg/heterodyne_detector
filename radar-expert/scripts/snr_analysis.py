#!/usr/bin/env python3
"""
SNR Analysis Utility: Measure signal-to-noise ratio improvement

Computes SNR before/after noise cancellation to validate beamformer effectiveness.
"""

import numpy as np
import torch


def compute_snr(signal, noise):
    """
    Compute signal-to-noise ratio in dB

    Args:
        signal: Complex signal tensor
        noise: Complex noise tensor

    Returns:
        SNR in dB
    """
    signal_power = torch.mean(torch.abs(signal) ** 2)
    noise_power = torch.mean(torch.abs(noise) ** 2)

    if noise_power == 0:
        return float('inf')

    snr_linear = signal_power / noise_power
    snr_db = 10 * torch.log10(snr_linear)

    return float(snr_db)


def estimate_signal_and_noise(rx_signal, known_signal=None, noise_window_start=None):
    """
    Estimate SNR from single received signal

    Two approaches:
    1. Known signal: Compute error vs. known clean reference
    2. Silence window: Use quiet period as noise estimate

    Args:
        rx_signal: Received signal (complex)
        known_signal: Clean reference signal (if available)
        noise_window_start: Sample index where noise-only period begins

    Returns:
        dict with 'snr', 'signal_power', 'noise_power'
    """

    if known_signal is not None:
        # Approach 1: Compare to known signal
        error = rx_signal - known_signal
        signal_power = torch.mean(torch.abs(known_signal) ** 2)
        noise_power = torch.mean(torch.abs(error) ** 2)

    elif noise_window_start is not None:
        # Approach 2: Use silence window for noise estimation
        noise_estimate = rx_signal[noise_window_start:]
        signal_estimate = rx_signal[:noise_window_start]

        signal_power = torch.mean(torch.abs(signal_estimate) ** 2)
        noise_power = torch.mean(torch.abs(noise_estimate) ** 2)

    else:
        raise ValueError("Provide either known_signal or noise_window_start")

    snr_db = 10 * torch.log10(signal_power / (noise_power + 1e-12))

    return {
        'snr_db': float(snr_db),
        'signal_power': float(signal_power),
        'noise_power': float(noise_power),
        'signal_power_dbm': float(10 * torch.log10(signal_power + 1e-12)),
        'noise_power_dbm': float(10 * torch.log10(noise_power + 1e-12))
    }


def compute_snr_improvement(before_cancellation, after_cancellation, reference=None):
    """
    Quantify SNR improvement from noise cancellation

    Args:
        before_cancellation: Signal before beamforming (RX1 or raw)
        after_cancellation: Signal after beamforming
        reference: Known clean signal (for error-based computation)

    Returns:
        dict with improvement metrics
    """

    if reference is not None:
        # SNR = signal power / error power
        error_before = before_cancellation - reference
        error_after = after_cancellation - reference

        signal_power = torch.mean(torch.abs(reference) ** 2)
        noise_before = torch.mean(torch.abs(error_before) ** 2)
        noise_after = torch.mean(torch.abs(error_after) ** 2)

        snr_before_db = 10 * torch.log10(signal_power / (noise_before + 1e-12))
        snr_after_db = 10 * torch.log10(signal_power / (noise_after + 1e-12))

    else:
        # Assumption: signal mostly unchanged, noise suppressed
        # Estimate from power reduction (rough)
        power_before = torch.mean(torch.abs(before_cancellation) ** 2)
        power_after = torch.mean(torch.abs(after_cancellation) ** 2)

        # If after power < before, assume signal preserved and noise reduced
        snr_before_db = 0.0  # Baseline
        snr_after_db = 10 * torch.log10(power_before / (power_after + 1e-12))

    improvement_db = snr_after_db - snr_before_db

    return {
        'snr_before_db': float(snr_before_db),
        'snr_after_db': float(snr_after_db),
        'improvement_db': float(improvement_db),
        'power_reduction_db': float(
            10 * torch.log10(
                torch.mean(torch.abs(before_cancellation) ** 2) /
                (torch.mean(torch.abs(after_cancellation) ** 2) + 1e-12)
            )
        )
    }


def estimate_noise_floor(signal, percentile=10):
    """
    Estimate noise floor from signal magnitude histogram

    Assumes lowest percentile is noise

    Args:
        signal: Complex signal
        percentile: Percentile for noise estimation (default 10th)

    Returns:
        Noise power (linear)
    """
    magnitudes = torch.abs(signal)
    noise_mag = torch.quantile(magnitudes, percentile / 100.0)
    noise_power = noise_mag ** 2

    return float(noise_power)


def signal_to_interference_ratio(signal_rx, interference_rx):
    """
    Compute SIR: Power of desired signal vs. interference

    Args:
        signal_rx: Channel with signal + interference
        interference_rx: Channel with interference reference (for estimation)

    Returns:
        SIR in dB
    """
    signal_power = torch.mean(torch.abs(signal_rx) ** 2)
    interference_power = torch.mean(torch.abs(interference_rx) ** 2)

    sir_db = 10 * torch.log10((signal_power + 1e-12) / (interference_power + 1e-12))

    return float(sir_db)


def analyze_beamformer_snr(rx1, rx2, beamformer_weights):
    """
    Analyze SNR characteristics of beamformer

    Args:
        rx1, rx2: Received signals
        beamformer_weights: Complex weights [w1, w2]

    Returns:
        SNR metrics dict
    """

    # Raw channel SNRs
    snr_rx1 = 10 * torch.log10(torch.mean(torch.abs(rx1) ** 2) + 1e-12)
    snr_rx2 = 10 * torch.log10(torch.mean(torch.abs(rx2) ** 2) + 1e-12)

    # Beamformed output
    beamformed = beamformer_weights[0] * rx1 + beamformer_weights[1] * rx2
    beamformed_power = torch.mean(torch.abs(beamformed) ** 2)

    # Noise analysis (assuming uncorrelated noise per channel)
    w1_mag = torch.abs(beamformer_weights[0])
    w2_mag = torch.abs(beamformer_weights[1])

    # Effective directivity
    directivity = (w1_mag + w2_mag) ** 2 / (w1_mag ** 2 + w2_mag ** 2)

    return {
        'snr_rx1_db': float(snr_rx1),
        'snr_rx2_db': float(snr_rx2),
        'beamformed_power_db': float(10 * torch.log10(beamformed_power + 1e-12)),
        'directivity_db': float(10 * torch.log10(directivity + 1e-12)),
        'weight_magnitudes': [float(w1_mag), float(w2_mag)],
        'weight_phases_deg': [
            float(torch.angle(beamformer_weights[0]) * 180 / np.pi),
            float(torch.angle(beamformer_weights[1]) * 180 / np.pi)
        ]
    }


if __name__ == "__main__":
    # Example usage
    print("SNR Analysis Utility")
    print("=" * 60)

    # Simulate signal and noise
    n_samples = 10000
    signal = 0.1 * torch.exp(1j * 2 * np.pi * 0.1 * torch.arange(n_samples) / 1000)
    noise = 0.01 * (torch.randn(n_samples) + 1j * torch.randn(n_samples))

    received = signal + noise

    print(f"Signal power: {torch.mean(torch.abs(signal)**2):.6f}")
    print(f"Noise power: {torch.mean(torch.abs(noise)**2):.6f}")

    snr_db = compute_snr(signal, noise)
    print(f"SNR: {snr_db:.2f} dB\n")

    # Beamformer simulation
    w = torch.tensor([0.7 + 0.1j, 0.7 - 0.1j])
    beamformed = w[0] * (signal + 0.5 * noise) + w[1] * (signal + 0.5 * noise)

    improvement = compute_snr_improvement(received, beamformed, reference=signal)
    print(f"SNR Improvement: {improvement['improvement_db']:.2f} dB")
