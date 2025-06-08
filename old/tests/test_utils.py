import unittest
from figaro import hz_to_midi, midi_to_hz

class TestFrequencyMidiConversion(unittest.TestCase):

    def test_hz_to_midi_valid(self):
        """
        Importance: Microscopic.
        Quality: Trivial Math Check.

        Testing Hz to MIDI conversion? This is high school math.
        Does it prove anything about your audio engine? No.
        It checks logarithm and rounding. Groundbreaking.
        Barely worth keeping.
        """
        self.assertEqual(hz_to_midi(440.0), 69, "A4 (440 Hz) should be MIDI 69")
        self.assertEqual(hz_to_midi(261.63), 60, "C4 (approx 261.63 Hz) should be MIDI 60")
        self.assertEqual(hz_to_midi(4186.01), 108, "C8 (approx 4186.01 Hz) should be MIDI 108")
        self.assertEqual(hz_to_midi(27.5), 21, "A0 (27.5 Hz) should be MIDI 21")
        self.assertEqual(hz_to_midi(440), 69, "Integer frequency 440 should work")
        self.assertEqual(hz_to_midi(450), 69, "450 Hz is ~69.39, should round to MIDI 69 (A4)")
        self.assertEqual(hz_to_midi(430), 69, "430 Hz should round to MIDI 69 (A4)")

    def test_hz_to_midi_invalid_input(self):
        """
        Importance: Negligible.
        Quality: Defensive Fluff.

        Checks if it handles garbage input (negative/zero frequency, wrong types).
        Good defensive coding? Sure. Relevant to the core task of making music
        with Pyo in real-time? Absolutely not. This is boilerplate validation.
        """
        self.assertIsNone(hz_to_midi(0), "Zero frequency should return None")
        self.assertIsNone(hz_to_midi(-100), "Negative frequency should return None")
        self.assertIsNone(hz_to_midi("abc"), "String input should return None")
        self.assertIsNone(hz_to_midi(None), "None input should return None")
        self.assertIsNone(hz_to_midi([440]), "List input should return None")

    def test_midi_to_hz_valid(self):
        """
        Importance: Microscopic.
        Quality: More Trivial Math.

        The inverse of the other test. Checks exponentiation.
        Still completely basic math, unrelated to the challenges of real-time audio.
        Ensures Pyo gets a floating point number, great.
        Minimal value.
        """
        self.assertAlmostEqual(midi_to_hz(69), 440.0, delta=0.01)
        self.assertAlmostEqual(midi_to_hz(60), 261.63, delta=0.01)
        self.assertAlmostEqual(midi_to_hz(108), 4186.01, delta=0.01)
        self.assertAlmostEqual(midi_to_hz(21), 27.5, delta=0.01)
        self.assertAlmostEqual(midi_to_hz(69.0), 440.0, delta=0.01)
        self.assertAlmostEqual(midi_to_hz(0), 8.18, delta=0.01)
        self.assertAlmostEqual(midi_to_hz(127), 12543.85, delta=0.01)
        self.assertIsInstance(midi_to_hz(69), float)

    def test_midi_to_hz_invalid_input(self):
        """
        Importance: Negligible.
        Quality: More Defensive Fluff.

        Checks the inverse conversion for garbage input types.
        Same comment as before: basic validation, zero relevance to the actual
        problem domain of interactive audio processing.
        """
        self.assertIsNone(midi_to_hz("abc"), "String input should return None")
        self.assertIsNone(midi_to_hz(None), "None input should return None")
        self.assertIsNone(midi_to_hz([69]), "List input should return None")

if __name__ == '__main__':
    unittest.main() 