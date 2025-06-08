import unittest
import logging

from figaro import GenerativeEngine

class TestGenerativeEngine(unittest.TestCase):

    def setUp(self):
        self.engine = GenerativeEngine()

    def test_generate_response_single_note_context(self):
        """
        Importance: Medium.
        Quality: Superficial.

        Okay, it checks if the *correct* frequency is *calculated* for the pluck synth
        when a note is detected. Does it check if this frequency is actually sent to
        the Pyo SoundEngine? Does it check if the 'pluck' synth sounds anything like
        a pluck or just a distorted mess? Does it test timing? No. Just dictionary checking.
        """
        midi_note = 65 # F4
        events = self.engine.generate_response(harmonic_context=midi_note)
        
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event['synth'], 'pad')
        self.assertEqual(event['action'], 'play_harmony')
        self.assertIn('midi_notes', event)
        self.assertEqual(event['midi_notes'], [midi_note])

    # --- Test for special A4 context --- 
    def test_generate_response_special_a4_context(self):
        """
        Importance: Low.
        Quality: Specific Rule Check.

        Tests the specific rule for MIDI note 69 (A4) generating [69, 76].
        Verifies the *logic*, not the sound.
        """
        midi_note = 69 # A4
        events = self.engine.generate_response(harmonic_context=midi_note)
        
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event['synth'], 'pad')
        self.assertEqual(event['action'], 'play_harmony')
        self.assertIn('midi_notes', event)
        self.assertEqual(event['midi_notes'], [69, 76]) # Check for A4 and E5

    def test_generate_response_single_note_invalid_midi(self):
        """
        Importance: Low.
        Quality: Edge Case Handling.

        Tests how the engine handles non-standard numerical inputs for harmonic_context.
        Specifically, it checks:
        - If a float input (e.g., 65.5) is correctly rounded to the nearest integer (66).
        - If the engine handles boundary MIDI values (0 and 127) without errors.
        This verifies the input processing logic for numerical contexts.
        """
        # midi_to_hz returns float or None. hz_to_midi returns int or None.
        # generate_response calls int(round(context)) internally first if context is numeric.
        
        # Case 1: Note is float but not integer-like
        input_midi_float = 65.5
        expected_midi_int = 66 # Should round up
        events_float = self.engine.generate_response(harmonic_context=input_midi_float)
        self.assertEqual(len(events_float), 1)
        event_float = events_float[0]
        self.assertEqual(event_float['synth'], 'pad')
        self.assertEqual(event_float['action'], 'play_harmony')
        self.assertEqual(event_float['midi_notes'], [expected_midi_int])

        # Test with an extremely high MIDI note (might be valid MIDI but unusual)
        test_midi_high = 127
        events_high = self.engine.generate_response(harmonic_context=test_midi_high)
        self.assertEqual(len(events_high), 1)
        event_high = events_high[0]
        self.assertEqual(event_high['midi_notes'], [test_midi_high])

        # Test with an extremely low MIDI note
        test_midi_low = 0
        events_low = self.engine.generate_response(harmonic_context=test_midi_low)
        self.assertEqual(len(events_low), 1)
        event_low = events_low[0]
        self.assertEqual(event_low['midi_notes'], [test_midi_low])

    def test_generate_response_unknown_context_type(self):
        """
        Importance: Microscopic.
        Quality: Defensive, but Pointless.

        Checks if it handles random Python types as context without crashing. Okay.
        Does this scenario *ever* happen in the real application? Probably not.
        Focus your testing effort on realistic scenarios and actual audio behaviour,
        not hypothetical type errors.
        """
        unknown_contexts = [[60, 64, 67], {'key': 0}, 1+2j]
        for context in unknown_contexts:
            with self.subTest(context=context):
                logging.debug(f"Testing unknown context type: {type(context)}")
                events = self.engine.generate_response(harmonic_context=context)
                self.assertEqual(events, [], f"Failed for context type: {type(context)}")

if __name__ == '__main__':
    unittest.main() 