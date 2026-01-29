import torch
import numpy as np


class RangeDopplerProcessor:
    """
    Range-Doppler processor for pseudo-pulse CW mode.
    
    Features:
    - Range FFT for range profile extraction
    - Doppler FFT for velocity estimation
    - CFAR detection for target identification
    - GPU-accelerated processing
    """
    
    def __init__(self, config, device=torch.device('cpu')):
        """
        Initialize Range-Doppler processor.
        
        Args:
            config: RANGE_DOPPLER config dict
            device: torch.device for GPU acceleration
        """
        self.config = config
        self.device = device
        
        # FFT parameters
        self.range_fft_size = config['range_fft_size']
        self.doppler_fft_size = config['doppler_fft_size']
        self.num_cpis = config['num_cpis']
        
        # Slow-time buffer for Doppler processing
        self.slow_time_buffer = torch.zeros(
            (self.num_cpis, self.range_fft_size),
            dtype=torch.complex64,
            device=device
        )
        self.buffer_idx = 0
        
        # CFAR parameters
        self.guard_cells = config['cfar_guard_cells']
        self.training_cells = config['cfar_training_cells']
        self.pfa = config['cfar_pfa']
        
        # Window for spectral leakage reduction
        if config['use_windowing']:
            self.window = torch.hann_window(
                self.range_fft_size,
                periodic=False,
                dtype=torch.float32,
                device=device
            )
        else:
            self.window = torch.ones(self.range_fft_size, device=device)
        
        # Compute CFAR threshold multiplier
        self._compute_cfar_threshold()
    
    def _compute_cfar_threshold(self):
        """
        Compute CFAR threshold multiplier based on Pfa.
        
        For CA-CFAR (Cell-Averaging), threshold = Pfa^(-1/N_training)
        """
        n_training = 2 * self.training_cells  # Guard cells on both sides
        self.cfar_threshold_mult = self.pfa ** (-1.0 / n_training)
    
    def _compute_range_profile(self, signal):
        """
        Compute range profile via FFT.
        
        Args:
            signal: Complex signal (N,)
            
        Returns:
            range_profile: FFT magnitude (range_fft_size,)
        """
        # Apply window
        windowed = signal[:self.range_fft_size] * self.window
        
        # Pad if needed
        if len(windowed) < self.range_fft_size:
            windowed = torch.cat([
                windowed,
                torch.zeros(self.range_fft_size - len(windowed), 
                           dtype=torch.complex64, device=self.device)
            ])
        
        # Range FFT
        range_fft = torch.fft.fft(windowed, n=self.range_fft_size)
        range_profile = torch.abs(range_fft)
        
        return range_profile
    
    def _cfar_detect(self, rd_map):
        """
        Cell-Averaging CFAR detection on Range-Doppler map.
        
        Args:
            rd_map: Range-Doppler magnitude map (doppler_bins, range_bins)
            
        Returns:
            detections: List of dicts with range_idx, doppler_idx, snr_db
        """
        detections = []
        
        # Convert to power
        power_map = rd_map ** 2
        
        # Iterate through cells (avoiding edges)
        for d_idx in range(self.doppler_fft_size // 2 - self.training_cells - self.guard_cells):
            for r_idx in range(self.range_fft_size // 2 - self.training_cells - self.guard_cells):
                # Test cell
                test_cell = power_map[d_idx + self.doppler_fft_size // 4, 
                                     r_idx + self.range_fft_size // 4]
                
                # Training cells (surrounding)
                train_start_d = d_idx + self.doppler_fft_size // 4 - self.training_cells - self.guard_cells
                train_end_d = d_idx + self.doppler_fft_size // 4 + self.training_cells + self.guard_cells + 1
                train_start_r = r_idx + self.range_fft_size // 4 - self.training_cells - self.guard_cells
                train_end_r = r_idx + self.range_fft_size // 4 + self.training_cells + self.guard_cells + 1
                
                # Guard cells (immediate neighbors to exclude)
                guard_start_d = d_idx + self.doppler_fft_size // 4 - self.guard_cells
                guard_end_d = d_idx + self.doppler_fft_size // 4 + self.guard_cells + 1
                guard_start_r = r_idx + self.range_fft_size // 4 - self.guard_cells
                guard_end_r = r_idx + self.range_fft_size // 4 + self.guard_cells + 1
                
                # Extract training cells (excluding guard cells)
                train_region = power_map[train_start_d:train_end_d, train_start_r:train_end_r].clone()
                train_region[guard_start_d-train_start_d:guard_end_d-train_start_d,
                            guard_start_r-train_start_r:guard_end_r-train_start_r] = 0
                
                # Threshold
                threshold = self.cfar_threshold_mult * torch.mean(train_region[train_region > 0] + 1e-10)
                
                # Detection
                if test_cell > threshold:
                    snr_db = 10 * torch.log10(test_cell / (torch.mean(train_region[train_region > 0]) + 1e-10))
                    detections.append({
                        'range_idx': r_idx,
                        'doppler_idx': d_idx,
                        'snr_db': float(snr_db.cpu().numpy()),
                        'power': float(test_cell.cpu().numpy())
                    })
        
        return detections
    
    def process(self, rx_signal):
        """
        Process single buffer through Range-Doppler pipeline.
        
        Args:
            rx_signal: Complex RX signal (N,)
            
        Returns:
            dict: Results with range_doppler_map, detections, etc.
        """
        # Step 1: Compute range profile
        range_profile = self._compute_range_profile(rx_signal)
        
        # Step 2: Update slow-time buffer (circular)
        self.slow_time_buffer[self.buffer_idx] = range_profile
        self.buffer_idx = (self.buffer_idx + 1) % self.num_cpis
        
        # Step 3: Doppler FFT (only if buffer is filled)
        if self.buffer_idx == 0:  # Buffer just wrapped (filled)
            # Doppler FFT across slow time
            rd_map_complex = torch.fft.fft(self.slow_time_buffer, dim=0, n=self.doppler_fft_size)
            rd_map = torch.fft.fftshift(torch.abs(rd_map_complex), dims=0)
            
            # Convert to dB
            rd_map_db = 20 * torch.log10(rd_map + 1e-10)
            
            # Step 4: CFAR detection
            detections = self._cfar_detect(rd_map)
            
            results = {
                'range_doppler_map': rd_map_db.cpu().numpy(),
                'range_doppler_complex': rd_map_complex.cpu().numpy(),
                'detections': detections,
                'buffer_full': True
            }
        else:
            # Buffer not yet full
            rd_map_db = torch.zeros((self.doppler_fft_size, self.range_fft_size))
            results = {
                'range_doppler_map': rd_map_db.cpu().numpy(),
                'range_doppler_complex': None,
                'detections': [],
                'buffer_full': False
            }
        
        return results
    
    def get_range_doppler_axes(self):
        """
        Get range and Doppler axes for plotting.
        
        Returns:
            ranges_m: Range axis in meters
            dopplers_hz: Doppler axis in Hz
        """
        max_range = self.config['max_range_m']
        max_doppler = self.config['max_doppler_hz']
        
        ranges_m = np.linspace(0, max_range, self.range_fft_size)
        dopplers_hz = np.linspace(-max_doppler, max_doppler, self.doppler_fft_size)
        
        return ranges_m, dopplers_hz
