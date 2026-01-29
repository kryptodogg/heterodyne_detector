
import unittest
import sys
import asyncio
from unittest.mock import MagicMock, patch

# Mock adi before importing sdr_interface
mock_adi = MagicMock()
sys.modules['adi'] = mock_adi

from sdr_interface import PlutoRadarInterface

class TestTxRxRouting(unittest.TestCase):
    def setUp(self):
        mock_adi.reset_mock()
        self.mock_sdr_instance = MagicMock()
        mock_adi.ad9361.return_value = self.mock_sdr_instance

    def test_routing_configuration(self):
        """Verify proper TX/RX routing and channel configuration"""

        # Initialize interface with simulate=False to trigger hardware code path
        # But since adi is mocked, it won't actually touch hardware
        sdr = PlutoRadarInterface(
            center_freq=2.4e9,
            sample_rate=10e6,
            simulate=False
        )

        # Run connect logic (using asyncio loop)
        asyncio.run(sdr.connect())

        # Verify ad9361 was initialized
        mock_adi.ad9361.assert_called_once()

        # Verify RX Configuration
        # We check the assignments to properties on the mock instance

        # Check RX channels
        self.assertEqual(self.mock_sdr_instance.rx_enabled_channels, [0, 1],
                         "RX channels 0 and 1 must be enabled")

        # Check TX channels
        self.assertEqual(self.mock_sdr_instance.tx_enabled_channels, [0, 1],
                         "TX channels 0 and 1 must be enabled")

        # Check Cyclic Buffer (Continuous Transmission)
        self.assertTrue(self.mock_sdr_instance.tx_cyclic_buffer,
                        "TX cyclic buffer must be enabled for continuous beacon")

        # Check Frequencies
        self.assertEqual(self.mock_sdr_instance.rx_lo, int(2.4e9))
        self.assertEqual(self.mock_sdr_instance.tx_lo, int(2.4e9))

        # Check Buffer Size
        self.assertEqual(self.mock_sdr_instance.rx_buffer_size, 2**16)

    def test_gain_configuration(self):
        """Verify gain control settings"""
        sdr = PlutoRadarInterface(simulate=False)
        asyncio.run(sdr.connect())

        # Check Manual Gain Mode
        self.assertEqual(self.mock_sdr_instance.gain_control_mode_chan0, 'manual')
        self.assertEqual(self.mock_sdr_instance.gain_control_mode_chan1, 'manual')

        # Check Gain Values
        self.assertEqual(self.mock_sdr_instance.rx_hardwaregain_chan0, 50)
        self.assertEqual(self.mock_sdr_instance.rx_hardwaregain_chan1, 50)

    def test_tx_pilot_tones(self):
        """Verify that pilot tones are sent to TX"""
        sdr = PlutoRadarInterface(simulate=False)
        asyncio.run(sdr.connect())

        # Check that sdr.tx() was called
        self.mock_sdr_instance.tx.assert_called_once()

        # Get the arguments passed to tx()
        args, _ = self.mock_sdr_instance.tx.call_args
        tx_data = args[0]

        # Verify it sent 2 channels of data
        self.assertEqual(len(tx_data), 2, "Must send data for 2 TX channels")

        # Verify data is complex
        import numpy as np
        self.assertTrue(np.iscomplexobj(tx_data[0]))
        self.assertTrue(np.iscomplexobj(tx_data[1]))

if __name__ == '__main__':
    unittest.main()
