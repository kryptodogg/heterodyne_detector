import unittest

try:
    from sdr_interface import PlutoRadarInterface
except Exception:
    PlutoRadarInterface = None


class TestSdrInterface(unittest.TestCase):
    def test_simulated_connect(self):
        if PlutoRadarInterface is None:
            self.skipTest("sdr_interface module not available in this environment")
        sdr = PlutoRadarInterface(simulate=True)
        ok = sdr.connect()
        self.assertTrue(ok, "Simulated SDR should connect successfully")


if __name__ == "__main__":
    unittest.main()
