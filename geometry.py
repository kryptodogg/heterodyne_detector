#!/usr/bin/env python3
"""
Geometry and Steering Vector Calculations for 2TX2RX Radar Array
Supports planar and linear array configurations
"""

import numpy as np
from typing import List, Tuple, Dict, Any
import json
from pathlib import Path


class ArrayGeometry:
    """
    2TX2RX array geometry for radar processing
    Handles position and orientation of antennas
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize array geometry from configuration

        Args:
            config: Dictionary containing geometry parameters
        """
        # Extract TX/RX positions
        self.tx1_pos = np.array(config["TX1"]["pos"])
        self.tx2_pos = np.array(config["TX2"]["pos"])
        self.rx1_pos = np.array(config["RX1"]["pos"])
        self.rx2_pos = np.array(config["RX2"]["pos"])

        # Extract orientations (Euler angles in degrees)
        self.tx1_orient = np.array(config["TX1"]["orient"])
        self.tx2_orient = np.array(config["TX2"]["orient"])
        self.rx1_orient = np.array(config["RX1"]["orient"])
        self.rx2_orient = np.array(config["RX2"]["orient"])

        # Convert orientations to rotation matrices
        self.tx1_rotation = self._euler_to_rotation_matrix(config["TX1"]["orient"])
        self.tx2_rotation = self._euler_to_rotation_matrix(config["TX2"]["orient"])
        self.rx1_rotation = self._euler_to_rotation_matrix(config["RX1"]["orient"])
        self.rx2_rotation = self._euler_to_rotation_matrix(config["RX2"]["orient"])

        # Array properties
        self.carrier_frequency = config.get("carrier_frequency", 2.45e9)
        self.array_type = config.get("array_type", "uniform_linear")

    def _euler_to_rotation_matrix(self, euler: np.ndarray) -> np.ndarray:
        """Convert Euler angles to rotation matrix (ZYX convention)"""
        roll, pitch, yaw = np.radians(euler)

        # Rotation matrix around X (roll)
        Rx = np.array(
            [
                [1, 0, 0],
                [0, np.cos(roll), -np.sin(roll)],
                [0, np.sin(roll), np.cos(roll)],
            ]
        )

        # Rotation matrix around Y (pitch)
        Ry = np.array(
            [
                [np.cos(pitch), 0, np.sin(pitch)],
                [0, 1, 0],
                [-np.sin(pitch), 0, np.cos(pitch)],
            ]
        )

        # Rotation matrix around Z (yaw)
        Rz = np.array(
            [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
        )

        # Combined rotation matrix
        return Rz @ Ry @ Rx

    def get_wavelength(self) -> float:
        """Get wavelength for current carrier frequency"""
        return 3e8 / self.carrier_frequency

    def get_array_vector(
        self, point_pos: np.ndarray, reference_pos: np.ndarray
    ) -> np.ndarray:
        """Get vector from reference to point"""
        return point_pos - reference_pos

    def get_antenna_normal(
        self, position: np.ndarray, orientation: np.ndarray
    ) -> np.ndarray:
        """Get antenna normal vector from position and orientation"""
        # Assume Z-axis is the primary pointing direction
        normal = orientation[:, 2]  # Z-axis of rotation matrix
        return normal / np.linalg.norm(normal)


class SteeringVectorCalculator:
    """
    Calculates steering vectors for 2TX2RX array
    Supports multiple steering directions and beamforming
    """

    def __init__(self, geometry: ArrayGeometry):
        """Initialize with array geometry"""
        self.geometry = geometry

    def compute_steering_vector(self, steering_angle: float) -> np.ndarray:
        """
        Compute steering vector for given angle

        Args:
            steering_angle: Steering angle in radians (0 = broadside, π = endfire)

        Returns:
            Complex steering vector for beamforming
        """
        # Use broadside array configuration
        # For 2TX2RX, steering vectors affect phase relationships
        wavelength = self.geometry.get_wavelength()

        # Default element spacing (can be customized)
        element_spacing = 0.062  # 62mm from your specifications

        # Phase shift for each element
        # For 2-element array: element 0 reference, element 1 has phase shift
        phase_shift = 2 * np.pi * element_spacing * np.sin(steering_angle) / wavelength

        # Steering vector (complex weights)
        steering_vector = np.array([1.0, np.exp(1j * phase_shift)], dtype=np.complex64)

        return steering_vector

    def compute_steering_matrix(self, steering_angles: List[float]) -> np.ndarray:
        """
        Compute steering matrix for multiple angles

        Args:
            steering_angles: List of steering angles in radians

        Returns:
            Matrix of steering vectors (n_angles, n_elements)
        """
        steering_vectors = []
        for angle in steering_angles:
            vector = self.compute_steering_vector(angle)
            steering_vectors.append(vector)

        return np.array(steering_vectors)

    def compute_doa_estimate(
        self, rx1: np.ndarray, rx2: np.ndarray
    ) -> Dict[str, float]:
        """
        Estimate Direction of Arrival using cross-correlation

        Args:
            rx1: Signal from first RX antenna
            rx2: Signal from second RX antenna

        Returns:
            Dictionary with DOA estimates in degrees
        """
        # Cross-correlation for time delay estimation
        correlation = np.correlate(rx1, rx2, mode="same")

        # Find peak to estimate time delay
        peak_idx = np.argmax(np.abs(correlation))
        max_correlation = np.abs(correlation[peak_idx])

        # Convert time delay to angle (simplified for planar array)
        # This is a basic approximation - more sophisticated methods exist
        array_length = self.geometry.get_array_vector(
            self.geometry.tx2_pos, self.geometry.tx1_pos
        )
        array_length = np.linalg.norm(array_length)

        # Simple DOA estimation
        time_delay_samples = peak_idx - len(correlation) // 2
        angle_rad = np.arcsin(np.clip(time_delay_samples / len(correlation), -1, 1))
        angle_deg = np.degrees(angle_rad)

        # Confidence based on correlation peak
        confidence = max_correlation / (np.max(np.abs(correlation)) + 1e-10)

        return {
            "angle_deg": angle_deg,
            "confidence": confidence,
            "peak_correlation": max_correlation,
            "time_delay_samples": time_delay_samples,
        }

    def get_baseline_vectors(self) -> np.ndarray:
        """
        Get baseline steering vectors for common directions

        Returns:
            Array of steering vectors for standard angles
        """
        # Standard directions: broadside, 30°, 60°, 90°, etc.
        baseline_angles = [0, np.pi / 6, np.pi / 3, np.pi / 2, 2 * np.pi / 3]

        baseline_vectors = []
        for angle in baseline_angles:
            vector = self.compute_steering_vector(angle)
            baseline_vectors.append(vector)

        return np.array(baseline_vectors)


class GeometryLoader:
    """
    Load and save array geometry configurations
    Supports JSON files and runtime modification
    """

    def __init__(self, default_config_path: str = "geometry.json"):
        """Initialize with default geometry file path"""
        self.default_config_path = Path(default_config_path)

    def load_geometry(self, config_path: str = None) -> ArrayGeometry:
        """
        Load geometry from JSON file

        Args:
            config_path: Path to geometry configuration file

        Returns:
            ArrayGeometry object
        """
        path = Path(config_path) if config_path else self.default_config_path

        if not path.exists():
            # Create default geometry if file doesn't exist
            default_geometry = self._create_default_geometry()
            self.save_geometry(default_geometry, path)
            return default_geometry

        try:
            with open(path, "r") as f:
                config = json.load(f)

            return ArrayGeometry(config)

        except Exception as e:
            raise RuntimeError(f"Failed to load geometry from {path}: {e}")

    def save_geometry(self, geometry: ArrayGeometry, config_path: str = None):
        """
        Save geometry configuration to JSON file

        Args:
            geometry: ArrayGeometry object to save
            config_path: Target file path
        """
        path = Path(config_path) if config_path else self.default_config_path

        config = {
            "TX1": {
                "pos": geometry.tx1_pos.tolist(),
                "orient": geometry.tx1_orient.tolist(),
            },
            "TX2": {
                "pos": geometry.tx2_pos.tolist(),
                "orient": geometry.tx2_orient.tolist(),
            },
            "RX1": {
                "pos": geometry.rx1_pos.tolist(),
                "orient": geometry.rx1_orient.tolist(),
            },
            "RX2": {
                "pos": geometry.rx2_pos.tolist(),
                "orient": geometry.rx2_orient.tolist(),
            },
            "carrier_frequency": geometry.carrier_frequency,
            "array_type": geometry.array_type,
        }

        # Create directory if needed
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(config, f, indent=2)

    def _create_default_geometry(self) -> ArrayGeometry:
        """
        Create default 2TX2RX planar array geometry
        Based on your specifications: 62mm element spacing
        """
        # Default configuration from your specifications
        config = {
            "TX1": {
                "pos": [0.0, 0.0, 0.0],
                "orient": [0.0, 0.0, 1.0],  # Pointing +Z
            },
            "TX2": {
                "pos": [0.062, 0.0, 0.0],  # 62mm to right
                "orient": [0.0, 0.0, 1.0],  # Pointing +Z
            },
            "RX1": {
                "pos": [-0.031, 0.0, 0.0],  # Halfway between TX
                "orient": [0.0, 0.0, 1.0],  # Pointing +Z
            },
            "RX2": {
                "pos": [-0.031, 0.0, 0.0],  # Halfway between TX
                "orient": [0.0, 0.0, 1.0],  # Pointing +Z
            },
            "carrier_frequency": 2.45e9,  # 2.45 GHz
            "array_type": "uniform_linear",
        }

        return ArrayGeometry(config)


# Utility functions for geometry operations
def compute_steering_weights(
    geometry: ArrayGeometry, steering_angles: List[float]
) -> Tuple[np.ndarray, List[float]]:
    """
    Compute steering weights for multiple angles

    Returns:
        Tuple of (weights, angles_degrees)
    """
    calculator = SteeringVectorCalculator(geometry)

    weights = []
    angles_deg = []

    for angle in steering_angles:
        weight = calculator.compute_steering_vector(angle)
        weights.append(weight)
        angles_deg.append(np.degrees(angle))

    return np.array(weights), angles_deg


def validate_array_geometry(geometry: ArrayGeometry) -> bool:
    """
    Validate array geometry configuration

    Returns:
        True if geometry is valid
    """
    # Check if all required fields are present
    required_fields = ["tx1_pos", "tx2_pos", "rx1_pos", "rx2_pos"]

    for field in required_fields:
        if not hasattr(geometry, field):
            return False

    # Check array type
    if geometry.array_type not in ["uniform_linear", "circular", "random"]:
        return False

    # Check carrier frequency
    if geometry.carrier_frequency <= 0:
        return False

    return True


# Default geometry for immediate use
DEFAULT_GEOMETRY = {
    "TX1": {"pos": [0.0, 0.0, 0.0], "orient": [0.0, 0.0, 1.0]},
    "TX2": {
        "pos": [0.062, 0.0, 0.0],  # 62mm spacing
        "orient": [0.0, 0.0, 1.0],
    },
    "RX1": {
        "pos": [-0.031, 0.0, 0.0],  # Halfway between TX
        "orient": [0.0, 0.0, 1.0],
    },
    "RX2": {
        "pos": [-0.031, 0.0, 0.0],  # Halfway between TX
        "orient": [0.0, 0.0, 1.0],
    },
    "carrier_frequency": 2.45e9,  # 2.45 GHz
    "array_type": "uniform_linear",
}
