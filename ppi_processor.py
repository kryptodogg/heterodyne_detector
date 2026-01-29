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
        self.steering_vectors = geometry.compute_steering_vector(angles_rad).to(device)

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
        # Ensure inputs are on the correct device
        rx1 = rx1.to(self.device)
        rx2 = rx2.to(self.device)

        # Select beamformer
        if self.beamformer == 'mvdr':
            beamform_fn = self._mvdr_beamform
        else:  # conventional
            beamform_fn = self._conventional_beamform

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
        self.steering_vectors = geometry.compute_steering_vector(angles_rad).to(device)

        # Pre-allocated buffers for performance
        self.ppi_map = torch.zeros((self.num_angles, self.num_range_bins),
                                   dtype=torch.float32, device=device)

        # Window for range processing
        self.window = torch.hann_window(self.num_range_bins, periodic=False,
                                       dtype=torch.float32, device=device)

    def process(self, rx1, rx2):
        """
        GPU-accelerated PPI processing with optimized operations.
        
        Args:
            rx1: RX1 complex signal (N,) - tensor on GPU
            rx2: RX2 complex signal (N,) - tensor on GPU

        Returns:
            dict: PPI results with map, angles, ranges
        """
        # Ensure inputs are on the correct device
        rx1 = rx1.to(self.device)
        rx2 = rx2.to(self.device)
        
        # Verify dimension ordering: (range_bins,) for single signal
        assert rx1.ndim == 1 and rx2.ndim == 1, "Input must be 1D tensors"

        # Select beamformer method
        if self.beamformer == 'mvdr':
            # Use the original MVDR implementation but optimized
            for i in range(self.num_angles):
                sv = self.steering_vectors[i]
                
                # Stack signals
                X = torch.stack([rx1, rx2], dim=0)  # Shape: (2, N)

                # Manual covariance R = X @ X^H / N
                N = X.shape[1]
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
                sv_rx = sv[:2].to(self.device)

                # MVDR weights: w = R_inv @ sv
                w1 = inv11 * sv_rx[0] + inv12 * sv_rx[1]
                w2 = inv21 * sv_rx[0] + inv22 * sv_rx[1]

                # Apply beamformer: y = w^H @ X
                beamformed = w1.conj() * X[0] + w2.conj() * X[1]

                # Compute range profile
                # Truncate to range bins
                signal_truncated = beamformed[:self.num_range_bins]

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

                # Store in PPI map
                self.ppi_map[i] = range_profile_db
        else:  # conventional beamformer
            # Vectorized conventional beamforming
            # Stack signals: shape (2, N)
            X = torch.stack([rx1, rx2], dim=0).to(self.device)
            
            # Get RX components of steering vectors: shape (num_angles, 2)
            sv_rx = self.steering_vectors[:, :2].conj().to(self.device)
            
            # Apply beamforming for all angles at once: (num_angles, 2) * (2, N) -> (num_angles, N)
            # Using broadcasting: multiply steering vectors with signals
            beamformed_all = torch.sum(sv_rx.unsqueeze(-1) * X.unsqueeze(0), dim=1)  # (num_angles, N)
            
            # Process all range profiles at once
            for i in range(self.num_angles):
                beamformed = beamformed_all[i]
                
                # Truncate to range bins
                signal_truncated = beamformed[:self.num_range_bins]

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

                # Store in PPI map
                self.ppi_map[i] = range_profile_db

        # Prepare output
        results = {
            'ppi_map': self.ppi_map.cpu().numpy(),
            'angles_deg': self.angles_deg,
            'ranges_m': np.linspace(0, self.max_range_m, self.num_range_bins)
        }

        return results