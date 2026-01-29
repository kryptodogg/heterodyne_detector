import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class Track:
    """
    Single target track with Kalman filter state.
    
    State: [x, y, vx, vy]^T (position and velocity)
    """
    track_id: int
    state: torch.Tensor  # [x, y, vx, vy]^T (4,)
    covariance: torch.Tensor  # (4, 4)
    age: int = 0
    hits: int = 1  # Number of measurements associated
    last_measurement: Optional[torch.Tensor] = None
    history: List[np.ndarray] = field(default_factory=list)
    
    def predict(self, F, Q, device):
        """
        Predict next state using constant velocity model.
        
        Args:
            F: State transition matrix (4, 4)
            Q: Process noise covariance (4, 4)
        """
        # Predict state: x = F @ x
        self.state = torch.matmul(F, self.state)
        
        # Predict covariance: P = F @ P @ F^T + Q
        self.covariance = torch.matmul(F, torch.matmul(self.covariance, F.T)) + Q
        
        self.age += 1
    
    def update(self, measurement, H, R, device):
        """
        Update state with measurement using Kalman update.
        
        Args:
            measurement: Measurement vector [x, y]^T (2,)
            H: Measurement matrix (2, 4)
            R: Measurement noise covariance (2, 2)
        """
        # Innovation: z - H @ x
        z_pred = torch.matmul(H, self.state)
        innovation = measurement - z_pred  # (2,)
        
        # Innovation covariance: S = H @ P @ H^T + R
        S = torch.matmul(H, torch.matmul(self.covariance, H.T)) + R
        
        # Kalman gain: K = P @ H^T @ S^-1
        try:
            S_inv = torch.linalg.inv(S)
            K = torch.matmul(self.covariance, torch.matmul(H.T, S_inv))
        except:
            # If inversion fails, skip update
            self.hits += 1
            return
        
        # Update state: x = x + K @ innovation
        self.state = self.state + torch.matmul(K, innovation)
        
        # Update covariance: P = (I - K @ H) @ P
        I = torch.eye(4, dtype=torch.float32, device=device)
        self.covariance = torch.matmul(I - torch.matmul(K, H), self.covariance)
        
        # Track statistics
        self.last_measurement = measurement.clone()
        self.hits += 1
        
        # Store position in history for trajectory visualization
        pos = self.state[:2].cpu().numpy()
        self.history.append(pos)
        if len(self.history) > 100:  # Keep last 100 positions
            self.history = self.history[-100:]
    
    def to_dict(self):
        """Convert track to dictionary for output."""
        return {
            'track_id': self.track_id,
            'x': float(self.state[0].item()),
            'y': float(self.state[1].item()),
            'vx': float(self.state[2].item()),
            'vy': float(self.state[3].item()),
            'speed': float(torch.sqrt(self.state[2]**2 + self.state[3]**2).item()),
            'age': self.age,
            'hits': self.hits,
            'confidence': min(1.0, self.hits / 5.0),  # Normalize to [0, 1]
            'history': np.array(self.history) if self.history else None
        }


class TargetTracker:
    """
    Multi-target tracker with Kalman filtering and nearest-neighbor data association.
    
    Features:
    - 4-state Kalman filter (x, y, vx, vy)
    - Nearest-neighbor data association with Mahalanobis distance gating
    - Track management (initialization, deletion)
    """
    
    def __init__(self, config, device=torch.device('cpu')):
        """
        Initialize target tracker.
        
        Args:
            config: TRACKING config dict
            device: torch.device for computation
        """
        self.config = config
        self.device = device
        
        # Track management
        self.max_tracks = config['max_tracks']
        self.gate_threshold = config['gate_threshold']
        self.track_timeout = config['track_timeout']
        self.min_hits_for_track = config.get('min_hits_for_track', 2)
        
        # Kalman filter matrices
        dt = config['dt']
        
        # State transition: x_k = F @ x_{k-1}
        self.F = torch.tensor([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=torch.float32, device=device)
        
        # Measurement matrix: z = H @ x
        self.H = torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=torch.float32, device=device)
        
        # Process noise covariance
        q = config['process_noise']
        self.Q = q * torch.eye(4, dtype=torch.float32, device=device)
        
        # Measurement noise covariance
        r = config['measurement_noise']
        self.R = r * torch.eye(2, dtype=torch.float32, device=device)
        
        # Initial uncertainty for new tracks
        self.initial_uncertainty = torch.eye(4, dtype=torch.float32, device=device)
        self.initial_uncertainty[2, 2] = config.get('init_velocity_uncertainty', 10.0) ** 2
        self.initial_uncertainty[3, 3] = config.get('init_velocity_uncertainty', 10.0) ** 2
        
        # Active tracks
        self.tracks: List[Track] = []
        self.next_track_id = 0
    
    def _compute_mahalanobis_distance(self, measurement, track):
        """
        Compute Mahalanobis distance between measurement and track.
        
        Args:
            measurement: [x, y]^T
            track: Track object
            
        Returns:
            distance: Mahalanobis distance (float)
        """
        # Predicted measurement
        z_pred = torch.matmul(self.H, track.state)
        
        # Innovation
        innovation = measurement - z_pred
        
        # Innovation covariance
        S = torch.matmul(self.H, torch.matmul(track.covariance, self.H.T)) + self.R
        
        # Mahalanobis distance: d = sqrt(innovation^T @ S^-1 @ innovation)
        try:
            S_inv = torch.linalg.inv(S)
            dist_sq = torch.matmul(innovation, torch.matmul(S_inv, innovation))
            distance = float(torch.sqrt(dist_sq + 1e-10).item())
        except:
            distance = float('inf')
        
        return distance
    
    def _nearest_neighbor_association(self, measurements):
        """
        Associate measurements to tracks using nearest neighbor with gating.
        
        Args:
            measurements: List of [x, y]^T tensors
            
        Returns:
            assignments: List of (track_idx, measurement_idx) tuples
        """
        assignments = []
        unassigned_measurements = set(range(len(measurements)))
        unassigned_tracks = set(range(len(self.tracks)))
        
        # For each track, find nearest measurement within gate
        for track_idx in list(unassigned_tracks):
            min_distance = self.gate_threshold
            best_measurement_idx = None
            
            for meas_idx in list(unassigned_measurements):
                distance = self._compute_mahalanobis_distance(
                    measurements[meas_idx], 
                    self.tracks[track_idx]
                )
                
                if distance < min_distance:
                    min_distance = distance
                    best_measurement_idx = meas_idx
            
            if best_measurement_idx is not None:
                assignments.append((track_idx, best_measurement_idx))
                unassigned_measurements.discard(best_measurement_idx)
                unassigned_tracks.discard(track_idx)
        
        return assignments, unassigned_measurements, unassigned_tracks
    
    def update(self, detections):
        """
        Update tracker with new detections.
        
        Args:
            detections: List of dicts with 'x', 'y' keys (from Range-Doppler or PPI)
            
        Returns:
            List of track dicts
        """
        # Convert detections to tensors
        measurements = []
        for det in detections:
            if 'x' in det and 'y' in det:
                meas = torch.tensor([det['x'], det['y']], dtype=torch.float32, device=self.device)
                measurements.append(meas)
        
        # Step 1: Predict all tracks
        for track in self.tracks:
            track.predict(self.F, self.Q, self.device)
        
        # Step 2: Data association
        if measurements:
            assignments, unassigned_meas, unassigned_tracks = \
                self._nearest_neighbor_association(measurements)
            
            # Update assigned tracks
            for track_idx, meas_idx in assignments:
                self.tracks[track_idx].update(
                    measurements[meas_idx],
                    self.H, self.R,
                    self.device
                )
            
            # Initialize new tracks from unassigned measurements
            for meas_idx in unassigned_meas:
                if len(self.tracks) < self.max_tracks:
                    # Initialize state with position and zero velocity
                    state = torch.zeros(4, dtype=torch.float32, device=self.device)
                    state[:2] = measurements[meas_idx]
                    
                    new_track = Track(
                        track_id=self.next_track_id,
                        state=state,
                        covariance=self.initial_uncertainty.clone(),
                        age=0,
                        hits=1
                    )
                    self.tracks.append(new_track)
                    self.next_track_id += 1
        
        # Step 3: Delete stale tracks
        self.tracks = [
            t for t in self.tracks 
            if t.age < self.track_timeout or t.hits >= self.min_hits_for_track
        ]
        
        # Step 4: Return confirmed tracks
        confirmed_tracks = [
            t.to_dict() for t in self.tracks
            if t.hits >= self.min_hits_for_track
        ]
        
        return confirmed_tracks
