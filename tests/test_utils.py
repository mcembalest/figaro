import unittest
import numpy as np
import logging
# Assuming figaro.py is in the parent directory or accessible via PYTHONPATH
from figaro import hz_to_midi, midi_to_hz

# Suppress logging during tests
# logging.disable(logging.CRITICAL)

class TestFrequencyMidiConversion(unittest.TestCase):

    def test_hz_to_midi_valid(self):
        """Test valid frequency to MIDI conversions."""
        self.assertEqual(hz_to_midi(440.0), 69, "A4 (440 Hz) should be MIDI 69")
        self.assertEqual(hz_to_midi(261.63), 60, "C4 (approx 261.63 Hz) should be MIDI 60")
        self.assertEqual(hz_to_midi(4186.01), 108, "C8 (approx 4186.01 Hz) should be MIDI 108")
        self.assertEqual(hz_to_midi(27.5), 21, "A0 (27.5 Hz) should be MIDI 21")
        # Test integer input
        self.assertEqual(hz_to_midi(440), 69, "Integer frequency 440 should work")
        # Test rounding
        self.assertEqual(hz_to_midi(450), 69, "450 Hz is ~69.39, should round to MIDI 69 (A4)")
        self.assertEqual(hz_to_midi(430), 69, "430 Hz should round to MIDI 69 (A4)") # A4 is 440, G#4 is 415.30. 430 is closer to A4.

    def test_hz_to_midi_invalid_input(self):
        """Test invalid inputs for hz_to_midi."""
        # Test zero frequency
        self.assertIsNone(hz_to_midi(0), "Zero frequency should return None")
        # Test negative frequency
        self.assertIsNone(hz_to_midi(-100), "Negative frequency should return None")
        # Test non-numeric types
        self.assertIsNone(hz_to_midi("abc"), "String input should return None")
        self.assertIsNone(hz_to_midi(None), "None input should return None")
        self.assertIsNone(hz_to_midi([440]), "List input should return None")
        # Check that warnings are logged (optional, requires checking log capture)
        # with self.assertLogs(level='WARNING') as log:
        #     hz_to_midi(-5)
        #     self.assertTrue(any("Invalid frequency" in msg for msg in log.output))

    def test_midi_to_hz_valid(self):
        """Test valid MIDI to frequency conversions."""
        self.assertAlmostEqual(midi_to_hz(69), 440.0, delta=0.01, msg="MIDI 69 should be approx 440.0 Hz")
        self.assertAlmostEqual(midi_to_hz(60), 261.63, delta=0.01, msg="MIDI 60 should be approx 261.63 Hz")
        self.assertAlmostEqual(midi_to_hz(108), 4186.01, delta=0.01, msg="MIDI 108 should be approx 4186.01 Hz")
        self.assertAlmostEqual(midi_to_hz(21), 27.5, delta=0.01, msg="MIDI 21 should be approx 27.5 Hz")
        # Test float MIDI input (function should handle it)
        self.assertAlmostEqual(midi_to_hz(69.0), 440.0, delta=0.01, msg="Float MIDI 69.0 should work")
        # Test MIDI 0
        self.assertAlmostEqual(midi_to_hz(0), 8.18, delta=0.01, msg="MIDI 0 should be approx 8.18 Hz")
        # Test MIDI 127
        self.assertAlmostEqual(midi_to_hz(127), 12543.85, delta=0.01, msg="MIDI 127 should be approx 12543.85 Hz")
        # Check return type is float
        self.assertIsInstance(midi_to_hz(69), float, "Return type should be float")

    def test_midi_to_hz_invalid_input(self):
        """Test invalid inputs for midi_to_hz."""
        # Test non-numeric types
        self.assertIsNone(midi_to_hz("abc"), "String input should return None")
        self.assertIsNone(midi_to_hz(None), "None input should return None")
        self.assertIsNone(midi_to_hz([69]), "List input should return None")
        # Note: The function doesn't explicitly check for MIDI range < 0 or > 127,
        # but it will calculate frequencies for them. Testing non-numeric is key.

if __name__ == '__main__':
    unittest.main() 