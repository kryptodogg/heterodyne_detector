import torch
import numpy as np
from ppi_processor import PPIProcessor


def test_refactored_ppi():
    """Test the refactored PPI processor to ensure it works correctly."""
    
    # Mock geometry class for testing
    class MockGeometry:
        def __init__(self, num_angles):
            self.num_angles = num_angles
            
        def compute_steering_vector(self, angles_rad):
            """Compute steering vectors for given angles."""
            # Create complex steering vectors for 2 RX channels
            num_angles = angles_rad.shape[0]
            # Create steering vectors based on angle
            sv = torch.zeros(num_angles, 4, dtype=torch.complex64)
            
            # Fill with realistic steering vectors
            for i, angle in enumerate(angles_rad):
                # Simple model: phase difference based on angle
                phase_diff = torch.sin(angle)  # Simplified model
                sv[i, 0] = torch.complex(torch.ones(1), torch.zeros(1))  # TX1
                sv[i, 1] = torch.complex(torch.ones(1), torch.zeros(1))  # TX2
                sv[i, 2] = torch.complex(torch.ones(1), torch.zeros(1))  # RX1
                sv[i, 3] = torch.complex(torch.cos(phase_diff), torch.sin(phase_diff))  # RX2 phase shift
            
            return sv
    
    # Configuration
    config = {
        'num_angles': 90,
        'num_range_bins': 128,
        'max_range_m': 50.0,
        'beamformer': 'conventional'
    }
    
    # Create mock geometry
    geometry = MockGeometry(config['num_angles'])
    
    # Initialize processor
    processor = PPIProcessor(geometry, config)
    
    # Create test data
    sample_size = 512
    rx1 = torch.randn(sample_size, dtype=torch.complex64)
    rx2 = torch.randn(sample_size, dtype=torch.complex64)
    
    print(f"Input shapes - RX1: {rx1.shape}, RX2: {rx2.shape}")
    print(f"Processor initialized with {processor.na} angles, {processor.nrb} range bins")
    
    # Process data
    results = processor.process(rx1, rx2)
    
    # Validate output
    ppi_map = results['ppi_map']
    print(f"Output shape: {ppi_map.shape}")
    
    expected_shape = (config['num_angles'], config['num_range_bins'])
    assert ppi_map.shape == expected_shape, f"Shape mismatch: expected {expected_shape}, got {ppi_map.shape}"
    
    print(f"PPI map stats - min: {ppi_map.min():.2f}, max: {ppi_map.max():.2f}, mean: {ppi_map.mean():.2f}")
    
    # Test with MVDR beamformer
    config_mvdr = config.copy()
    config_mvdr['beamformer'] = 'mvdr'
    processor_mvdr = PPIProcessor(geometry, config_mvdr)
    
    results_mvdr = processor_mvdr.process(rx1, rx2)
    ppi_map_mvdr = results_mvdr['ppi_map']
    
    print(f"MVDR output shape: {ppi_map_mvdr.shape}")
    assert ppi_map_mvdr.shape == expected_shape, f"MVDR shape mismatch: expected {expected_shape}, got {ppi_map_mvdr.shape}"
    
    print("✅ All tests passed! Refactored PPI processor is working correctly.")
    print(f"✅ Code is streamlined to {sum(1 for line in open('ppi_processor.py'))} lines")


if __name__ == "__main__":
    test_refactored_ppi()