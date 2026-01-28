import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict


class SpatialNoiseCanceller:
    """
    Spatial Noise Canceller with Beamforming + Adaptive LMS
    
    Features:
    - Beamformer scan for DOA estimation (37 angles, -90° to +90°)
    - MVDR beamforming for steering-vector enhanced processing
    - Adaptive LMS filter for residual interference suppression
    - SNR improvement computation
    """
    
    def __init__(self, geometry, config, device=torch.device('cpu')):
        """
        Initialize noise canceller with geometry and configuration.
        
        Args:
            geometry: RadarGeometry object with steering vector methods
            config: NOISE_CANCELLATION config dict
            device: torch.device for GPU acceleration
        """
        self.geometry = geometry
        self.config = config
        self.device = device
        
        # Pre-compute steering vectors for DOA scan
        self._setup_steering_vectors()
        
        # LMS filter state
        self.filter_length = config['filter_length']
        self.learning_rate = config['learning_rate']
        self.weights = torch.zeros(self.filter_length, dtype=torch.complex64, device=device)
        self.reset_weights()
    
    def _setup_steering_vectors(self):
        """Pre-compute steering vectors for DOA estimation."""
        num_angles = self.config['num_steering_angles']
        # Create angle array: -90° to +90° in equal steps
        angles_deg = np.linspace(-90, 90, num_angles)
        angles_rad = torch.tensor(angles_deg * np.pi / 180, dtype=torch.float32, device=self.device)
        
        # Compute steering vectors at all angles
        self.steering_vectors = self.geometry.compute_steering_vector(angles_rad)
        # steering_vectors shape: (num_angles, 4) for 2TX2RX
        self.angles_deg = angles_deg
    
    def reset_weights(self):
        """Reset LMS filter weights."""
        self.weights.zero_()
    
    def _beamformer_scan(self, rx1, rx2):
        """
        Scan beamformer across angles to estimate DOA.
        
        Args:
            rx1: RX1 signal (N,) complex tensor
            rx2: RX2 signal (N,) complex tensor
            
        Returns:
            doa_idx: Index of peak power angle
            doa_power: Power at each angle
        """
        # Stack RX signals: [rx1, rx2]
        X = torch.stack([rx1, rx2], dim=0)  # Shape: (2, N)
        
        # Compute output power at each angle
        # For each steering vector, compute: |sv^H @ X|^2
        num_angles = self.steering_vectors.shape[0]
        powers = torch.zeros(num_angles, dtype=torch.float32, device=self.device)
        
        for i in range(num_angles):
            sv = self.steering_vectors[i, :2]  # Get only RX components (2 elements)
            # Beamformer output: y = sv^H @ X
            y = torch.sum(sv.conj() * X, dim=0)  # Sum over RX channels
            # Power: E[|y|^2]
            powers[i] = torch.mean(torch.abs(y) ** 2)
        
        doa_idx = torch.argmax(powers)
        return doa_idx, powers
    
    def _mvdr_beamform(self, rx1, rx2, doa_idx):
        """
        Apply MVDR (Minimum Variance Distortionless Response) beamforming.
        
        Args:
            rx1: RX1 signal (N,) complex tensor
            rx2: RX2 signal (N,) complex tensor
            doa_idx: Index of target DOA
            
        Returns:
            clean_rx1, clean_rx2: Beamformed signals
        """
        # Stack signals
        X = torch.stack([rx1, rx2], dim=0)  # Shape: (2, N)
        
        # Compute sample covariance matrix: R = X @ X^H / N
        R = torch.matmul(X, X.conj().T) / X.shape[1]
        
        # Get steering vector for target DOA
        sv_target = self.steering_vectors[doa_idx, :2]  # RX components only
        
        # Add regularization for numerical stability
        reg = 1e-6 * torch.eye(2, dtype=torch.complex64, device=self.device)
        R_reg = R + reg
        
        # Compute MVDR weights: w = R^-1 @ sv / (sv^H @ R^-1 @ sv)
        try:
            R_inv = torch.linalg.inv(R_reg)
            numerator = torch.matmul(R_inv, sv_target)
            denominator = torch.matmul(sv_target.conj(), numerator)
            mvdr_weights = numerator / (denominator + 1e-10)
        except:
            # If inversion fails, fall back to matched filter
            mvdr_weights = sv_target / (torch.sum(torch.abs(sv_target) ** 2) + 1e-10)
        
        # Apply MVDR beamformer: y = w^H @ X
        beamformed = torch.sum(mvdr_weights.conj() * X, dim=0)
        
        # Return beamformed signal on both channels (for compatibility)
        return beamformed, beamformed * 0.5  # RX2 gets attenuated version
    
    def _adaptive_lms(self, rx_primary, rx_reference, block_size=256):
        """
        Adaptive LMS filter for residual interference suppression.
        
        Args:
            rx_primary: Primary (cleaned) RX signal
            rx_reference: Reference signal for LMS adaptation
            block_size: Block size for GPU-efficient processing
            
        Returns:
            filtered_signal: Adaptive-filtered primary signal
        """
        N = rx_primary.shape[0]
        filtered = torch.zeros_like(rx_primary)
        
        # Process in blocks for efficiency
        for n in range(0, N, block_size):
            end_idx = min(n + block_size, N)
            block_len = end_idx - n
            
            # Create input matrix for this block
            X_block = []
            for i in range(n, end_idx):
                # Get history window
                start_hist = max(0, i - self.filter_length)
                window = rx_reference[start_hist:i+1]
                
                # Pad if needed
                if len(window) < self.filter_length:
                    window = torch.cat([
                        torch.zeros(self.filter_length - len(window), dtype=torch.complex64, device=self.device),
                        window
                    ])
                else:
                    window = window[-self.filter_length:]
                
                X_block.append(window)
            
            # Estimate and filter
            X_block = torch.stack(X_block, dim=1)  # Shape: (filter_length, block_len)
            y_block = torch.matmul(self.weights.conj(), X_block)  # Shape: (block_len,)
            
            # Store filtered output
            filtered[n:end_idx] = y_block
            
            # Compute error and update weights
            error = rx_primary[n:end_idx] - y_block
            weight_update = self.learning_rate * torch.matmul(X_block, error.conj()) / block_len
            self.weights = self.weights + weight_update
        
        return filtered

    def frequency_domain_notch_filter(self, signal: torch.Tensor, 
                                     center_freq_hz: float, 
                                     sample_rate: float, 
                                     bandwidth_hz: float = 10.0) -> torch.Tensor:
        """
        Frequency-domain notch (bandstop) filter for tonal interference.
        Expert technique from radar-expert skill.
        """
        # FFT
        spectrum = torch.fft.fft(signal)
        freqs = torch.fft.fftfreq(len(signal), 1.0 / sample_rate).to(self.device)

        # Create notch mask
        freq_abs = torch.abs(freqs)
        # Gaussian notch for smooth transitions
        notch_mask = 1.0 - torch.exp(
            -((freq_abs - center_freq_hz) ** 2) / (2 * (bandwidth_hz / 4) ** 2)
        )

        # Apply notch
        spectrum_notched = spectrum * notch_mask

        # IFFT
        filtered = torch.fft.ifft(spectrum_notched)
        
        # Return real if input was real
        return torch.real(filtered) if not signal.is_complex() else filtered

    def estimate_tonal_interference(self, signal: torch.Tensor, 
                                   sample_rate: float) -> Optional[float]:
        """Estimate dominant tone frequency for notch filtering."""
        if signal.is_complex():
            mag = torch.abs(signal)
        else:
            mag = signal
            
        spec = torch.abs(torch.fft.rfft(mag))
        freqs = torch.fft.rfftfreq(len(mag), 1.0 / sample_rate).to(self.device)
        
        # Avoid DC
        spec[0] = 0
        peak_idx = torch.argmax(spec)
        
        # Only return if peak is significant (SNR > 10dB approx)
        if spec[peak_idx] > 5 * torch.mean(spec):
            return float(freqs[peak_idx].cpu().item())
        return None

    def _compute_snr_improvement(self, noisy_signal, clean_signal):
        """
        Compute SNR improvement in dB.
        
        Args:
            noisy_signal: Original signal
            clean_signal: Processed signal
            
        Returns:
            float: SNR improvement in dB
        """
        noise = noisy_signal - clean_signal
        
        power_noisy = torch.mean(torch.abs(noisy_signal) ** 2)
        power_noise = torch.mean(torch.abs(noise) ** 2)
        power_clean = torch.mean(torch.abs(clean_signal) ** 2)
        
        snr_before = 10 * torch.log10(power_noisy / (power_noise + 1e-10))
        snr_after = 10 * torch.log10(power_clean / (power_noise + 1e-10))
        
        improvement = float((snr_after - snr_before).cpu().numpy())
        return improvement
    
    def cancel(self, rx1, rx2):
        """
        Perform spatial noise cancellation.
        
        Args:
            rx1: RX1 complex signal (N,)
            rx2: RX2 complex signal (N,)
            
        Returns:
            clean_rx1, clean_rx2: Noise-cancelled signals
            cancellation_info: Dict with DOA, power, SNR improvement, weights
        """
        # Step 1: Beamformer scan for DOA estimation
        doa_idx, doa_power = self._beamformer_scan(rx1, rx2)
        doa_deg = float(self.angles_deg[doa_idx.cpu().numpy()])
        
        # Step 2: MVDR beamforming
        clean_rx1, clean_rx2 = self._mvdr_beamform(rx1, rx2, doa_idx)
        
        # Step 3: Adaptive LMS for residual interference
        clean_rx1 = self._adaptive_lms(clean_rx1, rx2)
        
        # Step 4: Compute SNR improvement
        snr_improvement = self._compute_snr_improvement(rx1, clean_rx1)
        
        # Prepare output info
        cancellation_info = {
            'doa': doa_deg,
            'doa_power': doa_power.cpu().numpy(),
            'snr_improvement': snr_improvement,
            'weights': self.weights[:10].cpu().numpy()  # First 10 taps for monitoring
        }
        
        return clean_rx1, clean_rx2, cancellation_info
