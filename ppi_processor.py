import torch
import numpy as np


class PPIProcessor:
    """
    PPI (Plan Position Indicator) processor for polar radar display.

    Features:
    - Digital beamforming at multiple angles
    - Range profile extraction at each angle
    - GPU-accelerated beamforming
    """

    def __init__(self, geometry, config, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        """
        Initialize PPI processor.

        Args:
            geometry: RadarGeometry object with steering vector methods
            config: PPI_CONFIG dict
            device: torch.device for GPU acceleration
        """
        self.geometry = geometry
        self.config = config
        self.device = device

        # Beamforming parameters
        self.num_angles = config['num_angles']
        self.num_range_bins = config['num_range_bins']
        self.max_range_m = config['max_range_m']
        self.beamformer = config.get('beamformer', 'conventional')

        # Generate angle array and steering vectors
        self.angles_deg = np.linspace(-90, 90, self.num_angles)
        angles_rad = torch.tensor(self.angles_deg * np.pi / 180, dtype=torch.float32, device=device)
        self.steering_vectors = geometry.compute_steering_vector(angles_rad)

        # PPI map storage
        self.ppi_map = torch.zeros((self.num_angles, self.num_range_bins),
                                   dtype=torch.float32, device=device)

        # Window for range processing
        self.window = torch.hann_window(self.num_range_bins, periodic=False,
                                       dtype=torch.float32, device=device)

    def _conventional_beamform(self, rx1, rx2, steering_vector):
        """
        Conventional (Delay-and-Sum) beamformer.

        Args:
            rx1: RX1 signal (N,) complex tensor
            rx2: RX2 signal (N,) complex tensor
            steering_vector: Steering vector for target angle (4,)

        Returns:
            beamformed: Beamformed signal (N,)
        """
        # Stack signals
        X = torch.stack([rx1, rx2], dim=0).to(self.device)  # Shape: (2, N)

        # Extract RX components
        sv = steering_vector[:2].conj().to(self.device)

        # Beamform: y = sum(sv^H * X)
        beamformed = torch.sum(sv.unsqueeze(1) * X, dim=0)

        return beamformed

    def _mvdr_beamform(self, rx1, rx2, steering_vector):
        """
        MVDR (Minimum Variance Distortionless Response) beamformer.
        Manual 2x2 implementation to bypass hipblas issues.
        """
        # Stack signals
        X = torch.stack([rx1, rx2], dim=0).to(self.device)  # Shape: (2, N)
        N = X.shape[1]

        # Manual covariance R = X @ X^H / N
        r11 = torch.sum(X[0] * X[0].conj()) / N
        r12 = torch.sum(X[0] * X[1].conj()) / N
        r21 = r12.conj()
        r22 = torch.sum(X[1] * X[1].conj()) / N

        # Regularization
        reg = 1e-6
        r11 += reg
        r22 += reg

        # 2x2 Inverse: 1/det * [[d, -b], [-c, a]]
        det = r11 * r22 - r12 * r21
        inv11 = r22 / det
        inv12 = -r12 / det
        inv21 = -r21 / det
        inv22 = r11 / det

        # Extract steering vector
        sv = steering_vector[:2].to(self.device)

        # MVDR weights: w = R_inv @ sv
        w1 = inv11 * sv[0] + inv12 * sv[1]
        w2 = inv21 * sv[0] + inv22 * sv[1]

        # Apply beamformer: y = w^H @ X
        beamformed = w1.conj() * X[0] + w2.conj() * X[1]
        return beamformed

    def _compute_range_profile(self, signal):
        """
        Compute range profile via FFT.

        Args:
            signal: Complex signal (N,)

        Returns:
            range_profile: FFT magnitude (num_range_bins,)
        """
        # Truncate to range bins
        signal_truncated = signal[:self.num_range_bins]

        # Pad if needed
        if len(signal_truncated) < self.num_range_bins:
            signal_truncated = torch.cat([
                signal_truncated,
                torch.zeros(self.num_range_bins - len(signal_truncated),
                           dtype=torch.complex64, device=self.device)
            ])

        # Apply window and FFT
        windowed = signal_truncated * self.window
        range_fft = torch.fft.fft(windowed, n=self.num_range_bins)
        range_profile = torch.abs(range_fft)

        # Convert to dB
        range_profile_db = 20 * torch.log10(range_profile + 1e-10)

        return range_profile_db

    def process(self, rx1, rx2):
        """
        Process signal through beamformer to generate PPI map.

        Args:
            rx1: RX1 complex signal (N,)
            rx2: RX2 complex signal (N,)

        Returns:
            dict: PPI results with map, angles, ranges
        """
        # Select beamformer
        if self.beamformer == 'mvdr':
            beamform_fn = self._mvdr_beamform
        else:  # conventional
            beamform_fn = self._conventional_beamform

        # Ensure inputs are on the correct device
        rx1 = rx1.to(self.device)
        rx2 = rx2.to(self.device)

        # Beamform at each angle
        for i in range(self.num_angles):
            # Get steering vector for this angle
            sv = self.steering_vectors[i]

            # Beamform
            beamformed = beamform_fn(rx1, rx2, sv)

            # Compute range profile
            self.ppi_map[i] = self._compute_range_profile(beamformed)

        # Prepare output
        results = {
            'ppi_map': self.ppi_map.cpu().numpy(),
            'angles_deg': self.angles_deg,
            'ranges_m': np.linspace(0, self.max_range_m, self.num_range_bins)
        }

        return results


class OptimizedPPIProcessor:
    """
    Highly optimized PPI processor with GPU memory management and batch processing.
    """
    
    def __init__(self, geometry, config, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        """
        Initialize optimized PPI processor.

        Args:
            geometry: RadarGeometry object with steering vector methods
            config: PPI_CONFIG dict
            device: torch.device for GPU acceleration
        """
        self.device = device
        self.geometry = geometry
        self.config = config

        # Beamforming parameters
        self.num_angles = config['num_angles']
        self.num_range_bins = config['num_range_bins']
        self.max_range_m = config['max_range_m']
        self.beamformer = config.get('beamformer', 'conventional')

        # Generate angle array and steering vectors
        self.angles_deg = np.linspace(-90, 90, self.num_angles)
        angles_rad = torch.tensor(self.angles_deg * np.pi / 180, dtype=torch.float32, device=device)
        self.steering_vectors = geometry.compute_steering_vector(angles_rad)

        # Pre-allocated buffers for performance
        self.ppi_map = torch.zeros((self.num_angles, self.num_range_bins),
                                   dtype=torch.float32, device=device)

        # Window for range processing
        self.window = torch.hann_window(self.num_range_bins, periodic=False,
                                       dtype=torch.float32, device=device)
        
        # Batch processing buffer
        self.batch_buffer = None

    def _conventional_beamform_batch(self, rx1_batch, rx2_batch):
        """
        Batch conventional beamforming for all angles at once.

        Args:
            rx1_batch: RX1 signals (batch_size, N)
            rx2_batch: RX2 signals (batch_size, N)

        Returns:
            beamformed_batch: Beamformed signals (batch_size, num_angles, N)
        """
        batch_size = rx1_batch.shape[0]
        num_signals = rx1_batch.shape[1]
        
        # Expand signals to match angles
        rx1_expanded = rx1_batch.unsqueeze(1).expand(-1, self.num_angles, -1)  # (batch, angles, N)
        rx2_expanded = rx2_batch.unsqueeze(1).expand(-1, self.num_angles, -1)  # (batch, angles, N)
        
        # Expand steering vectors to match batch
        sv_expanded = self.steering_vectors.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, angles, 4)
        
        # Extract RX components of steering vectors
        sv_rx = sv_expanded[:, :, :2].conj()  # (batch, angles, 2)
        
        # Stack rx signals
        X = torch.stack([rx1_expanded, rx2_expanded], dim=2)  # (batch, angles, 2, N)
        
        # Apply beamforming: sum(sv * X) for each angle
        beamformed = torch.sum(sv_rx.unsqueeze(-1) * X, dim=2)  # (batch, angles, N)
        
        return beamformed

    def _mvdr_beamform_batch(self, rx1_batch, rx2_batch):
        """
        Batch MVDR beamforming for all angles at once.
        """
        batch_size = rx1_batch.shape[0]
        N = rx1_batch.shape[1]
        
        # Stack signals
        X = torch.stack([rx1_batch, rx2_batch], dim=2)  # (batch_size, N, 2)
        
        # Compute covariance matrices for each batch
        # R = X^H @ X / N
        X_H = torch.conj(X.transpose(1, 2))  # (batch_size, 2, N)
        R = torch.bmm(X_H, X) / N  # (batch_size, 2, 2)
        
        # Add regularization
        reg = 1e-6
        I_reg = torch.eye(2, device=self.device, dtype=torch.float32) * reg
        R = R + I_reg
        
        # Compute inverse using torch.inverse
        R_inv = torch.inverse(R)  # (batch_size, 2, 2)
        
        # Apply MVDR for each angle
        sv_rx = self.steering_vectors[:, :2].unsqueeze(0).expand(batch_size, -1, -1)  # (batch, angles, 2)
        
        # Compute weights: w = R_inv @ sv
        weights = torch.bmm(R_inv.unsqueeze(1).expand(-1, self.num_angles, -1, -1), 
                            sv_rx.unsqueeze(-1)).squeeze(-1)  # (batch, angles, 2)
        
        # Apply weights to signals
        X_stacked = X.transpose(1, 2)  # (batch, 2, N)
        beamformed = torch.sum(weights.unsqueeze(-1) * X_stacked.unsqueeze(1), dim=2)  # (batch, angles, N)
        
        return beamformed

    def _compute_range_profiles_batch(self, signals_batch):
        """
        Compute range profiles for a batch of signals.

        Args:
            signals_batch: Complex signals (batch_size, num_angles, N)

        Returns:
            range_profiles: FFT magnitudes (batch_size, num_angles, num_range_bins)
        """
        batch_size, num_angles, N = signals_batch.shape
        
        # Truncate/pad signals to range bins
        if N < self.num_range_bins:
            # Pad signals
            padded_signals = torch.cat([
                signals_batch,
                torch.zeros(batch_size, num_angles, self.num_range_bins - N,
                           dtype=signals_batch.dtype, device=self.device)
            ], dim=2)
        elif N > self.num_range_bins:
            # Truncate signals
            padded_signals = signals_batch[:, :, :self.num_range_bins]
        else:
            padded_signals = signals_batch
        
        # Apply window
        windowed = padded_signals * self.window  # Broadcasting: (batch, angles, num_range_bins)
        
        # Apply FFT
        range_fft = torch.fft.fft(windowed, n=self.num_range_bins, dim=2)
        range_profile = torch.abs(range_fft)
        
        # Convert to dB
        range_profile_db = 20 * torch.log10(range_profile + 1e-10)
        
        return range_profile_db

    def process(self, rx1, rx2):
        """
        Process signal through beamformer to generate PPI map using GPU-first operations.

        Args:
            rx1: RX1 complex signal (N,) or (batch_size, N)
            rx2: RX2 complex signal (N,) or (batch_size, N)

        Returns:
            dict: PPI results with map, angles, ranges
        """
        # Ensure inputs are on the correct device
        rx1 = rx1.to(self.device)
        rx2 = rx2.to(self.device)
        
        # Handle single signal case
        if rx1.dim() == 1:
            rx1 = rx1.unsqueeze(0)  # Add batch dimension
            rx2 = rx2.unsqueeze(0)  # Add batch dimension
            single_input = True
        else:
            single_input = False

        # Select beamformer
        if self.beamformer == 'mvdr':
            beamformed_batch = self._mvdr_beamform_batch(rx1, rx2)
        else:  # conventional
            beamformed_batch = self._conventional_beamform_batch(rx1, rx2)

        # Compute range profiles
        range_profiles = self._compute_range_profiles_batch(beamformed_batch)

        # Extract results (first batch if single input)
        if single_input:
            self.ppi_map = range_profiles[0]  # (num_angles, num_range_bins)
        else:
            # For batch processing, return all results
            self.ppi_map = range_profiles  # (batch_size, num_angles, num_range_bins)

        # Prepare output
        if single_input:
            results = {
                'ppi_map': self.ppi_map.cpu().numpy(),
                'angles_deg': self.angles_deg,
                'ranges_m': np.linspace(0, self.max_range_m, self.num_range_bins)
            }
        else:
            results = {
                'ppi_maps': self.ppi_map.cpu().numpy(),
                'angles_deg': self.angles_deg,
                'ranges_m': np.linspace(0, self.max_range_m, self.num_range_bins)
            }

        return results