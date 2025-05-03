import unittest
from unittest.mock import MagicMock, patch
import logging

# Assuming figaro.py is in the parent directory or accessible via PYTHONPATH
from figaro import GenerativeEngine, midi_to_hz, PITCH_CLASSES

# Suppress logging during tests
# logging.disable(logging.CRITICAL) # REMOVED

class TestGenerativeEngine(unittest.TestCase):

    def setUp(self):
        self.engine = GenerativeEngine()
        # Clear cache before each test
        self.engine.chord_notes_cache = {}

    # --- Tests for _get_chord_midi_notes --- 

    def test_get_chord_midi_notes_valid(self):
        """
        Importance: Low.
        Quality: Trivial.

        Checks if your little helper function can count intervals. Congratulations.
        Does this verify that Pyo will actually play these notes correctly, or that
        they form a musically sensible chord in the target octave? Absolutely not.
        It's basic arithmetic checking, nothing to do with audio.
        """
        # C Major (Root PC = 0)
        expected_c_maj = [48, 52, 55] # C4, E4, G4 (TARGET_OCTAVE=4)
        self.assertEqual(self.engine._get_chord_midi_notes('C_maj'), expected_c_maj)
        
        # G# Minor (Root PC = 8)
        expected_gs_min = [56, 59, 63] # G#4, B4, D#5 
        self.assertEqual(self.engine._get_chord_midi_notes('G#_min'), expected_gs_min)
        
        # B Minor (Root PC = 11)
        expected_b_min = [59, 62, 66] # B4, D5, F#5
        self.assertEqual(self.engine._get_chord_midi_notes('B_min'), expected_b_min)

    def test_get_chord_midi_notes_invalid_format(self):
        """
        Importance: Microscopic.
        Quality: Pedantic.

        Tests if your parser correctly rejects garbage input. Okay, fine.
        This prevents maybe one category of dumb errors, but does zero for testing
        the actual *sound* generation or its musicality. Barely registers as a test.
        """
        invalid_names = [None, '', 'Cmaj', 'C_', '_maj', 'C_major', 'X_maj']
        for name in invalid_names:
            with self.subTest(name=name):
                self.assertIsNone(self.engine._get_chord_midi_notes(name), f"Failed for: {name}")

    def test_get_chord_midi_notes_unknown_quality(self):
        """
        Importance: Microscopic.
        Quality: Also Pedantic.

        More garbage input checking. At least it's consistent with the *actual* code
        now (rejecting unknown qualities). Still, utterly unrelated to audio output.
        Who cares if it rejects 'C_dim' if it can't even prove 'C_maj' sounds right?
        """
        # Expect None now due to stricter parsing in _get_chord_midi_notes
        self.assertIsNone(self.engine._get_chord_midi_notes('C_dim'))
        self.assertIsNone(self.engine._get_chord_midi_notes('C_unknown'))
        # ---- Original Test ----
        # Expect C Major notes
        # expected_c_maj = [48, 52, 55] 
        # Check with logging enabled might show a warning here
        # self.assertEqual(self.engine._get_chord_midi_notes('C_dim'), expected_c_maj)
        # self.assertEqual(self.engine._get_chord_midi_notes('C_unknown'), expected_c_maj)
        # ---------------------

    def test_get_chord_midi_notes_caching(self):
        """
        Importance: Negligible.
        Quality: Pointless.

        Tests if a Python dictionary works as a cache. Wow. Are you testing Python itself?
        This has *nothing* to do with Pyo, audio, real-time performance, or music.
        This is the kind of test junior developers write to feel productive. Delete it.
        """
        chord_name = 'D_min'
        expected_d_min = [50, 53, 57] # D4, F4, A4
        
        # First call
        result1 = self.engine._get_chord_midi_notes(chord_name)
        self.assertEqual(result1, expected_d_min)
        self.assertIn(chord_name, self.engine.chord_notes_cache)
        self.assertEqual(self.engine.chord_notes_cache[chord_name], expected_d_min)
        
        # Modify cache to check if it's reused
        self.engine.chord_notes_cache[chord_name] = [0, 0, 0] 
        
        # Second call should return the cached (modified) value
        result2 = self.engine._get_chord_midi_notes(chord_name)
        self.assertEqual(result2, [0, 0, 0])

    # --- Tests for generate_response --- 

    def test_generate_response_no_context(self):
        """
        Importance: Low.
        Quality: Obvious.

        Tests that if you give it nothing, it does nothing. Shocking.
        Minimal value, just confirms the basic guard condition. Doesn't test audio.
        """
        events = self.engine.generate_response(beat_phase=None, harmonic_context=None, bpm=None)
        self.assertEqual(events, [])

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
        expected_freq = midi_to_hz(midi_note)
        events = self.engine.generate_response(beat_phase=0, harmonic_context=midi_note, bpm=120)
        
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event['synth'], 'pluck')
        self.assertEqual(event['action'], 'trigger')
        self.assertAlmostEqual(event['freq'], expected_freq, places=5)

    def test_generate_response_single_note_invalid_midi(self):
        """
        Importance: Low.
        Quality: Needlessly Complex Mocking.

        You're testing edge cases around MIDI conversion? Fine, but the way you do it
        by patching `midi_to_hz` is just more disconnected mocking.
        It verifies your internal logic handles a None return, but doesn't prove anything
        about how Pyo or the *real* `midi_to_hz` behaves with weird inputs.
        Focus on testing the core functionality, not every conceivable numerical edge case.
        """
        # midi_to_hz returns float or None. hz_to_midi returns int or None.
        # generate_response calls hz_to_midi internally first if context is numeric.
        # Test cases where hz_to_midi would fail inside the function are tricky.
        # Let's test where the input midi note itself is bad *before* conversion.
        
        # Case 1: Note is float but not integer-like
        # midi_to_hz should calculate frequency based on the float value
        input_midi_float = 65.5
        events_float = self.engine.generate_response(beat_phase=0, harmonic_context=input_midi_float, bpm=120)
        # Expected frequency is midi_to_hz(65.5)
        expected_freq_float = midi_to_hz(input_midi_float) 
        self.assertEqual(len(events_float), 1)
        self.assertAlmostEqual(events_float[0]['freq'], expected_freq_float, places=5, 
                             msg=f"Freq for {input_midi_float} should be approx {expected_freq_float:.5f}")

        # Case 2: Simulate midi_to_hz returning None
        # We patch this specifically for the second part of the test
        with patch('figaro.midi_to_hz', return_value=None) as mock_midi_hz_fails:
            invalid_midi_note = 65 # Choose a note that would normally work
            events_none = self.engine.generate_response(beat_phase=0.0, harmonic_context=invalid_midi_note, bpm=120)
            
            # Verify midi_to_hz was called
            mock_midi_hz_fails.assert_called_once_with(invalid_midi_note)
            
            # Verify that NO events are generated because frequency conversion failed
            self.assertEqual(events_none, [], "No events should be generated when midi_to_hz returns None")

    @patch('figaro.GenerativeEngine._get_chord_midi_notes')
    def test_generate_response_chord_context_valid(self, mock_get_notes):
        """
        Importance: Medium.
        Quality: More Superficial Mocking.

        Same story as the single note test. Checks if the *root* frequency is calculated
        correctly based on a mocked chord lookup. Doesn't verify the sound engine gets called,
        doesn't verify the sound itself, doesn't test timing. Are you *trying* to avoid testing
        the actual audio part?
        """
        chord_name = 'A_min'
        # Correct root note for A minor with TARGET_OCTAVE=4 is A4 = MIDI 69
        root_note_midi = 69 
        # Configure the mock to return the expected notes for A_min
        # We don't strictly need the full list, just the root for this test, 
        # but it's good practice to return what the real function would.
        mock_get_notes.return_value = [69, 72, 76] # A4, C5, E5
        
        expected_root_freq = midi_to_hz(root_note_midi)
        
        events = self.engine.generate_response(beat_phase=0.5, harmonic_context=chord_name, bpm=100)
        
        # Verify the mock was called
        mock_get_notes.assert_called_once_with(chord_name)
        
        # Assertions on the event
        self.assertEqual(len(events), 1, "Expected one event for chord context")
        event = events[0]
        self.assertEqual(event['synth'], 'pluck')
        self.assertEqual(event['action'], 'trigger')
        self.assertAlmostEqual(event['freq'], expected_root_freq, places=5)

    def test_generate_response_chord_context_invalid(self):
        """
        Importance: Microscopic.
        Quality: Repetitive Pedantry.

        More testing of invalid inputs to the chord parser, indirectly through the main
        response function. Adds almost zero value over the direct parser tests.
        Still completely ignores the audio output.
        """
        invalid_names = [None, '', 'Cmaj', 'C_', '_maj']
        for name in invalid_names:
            with self.subTest(name=name):
                events = self.engine.generate_response(beat_phase=0, harmonic_context=name, bpm=120)
                self.assertEqual(events, [], f"Failed for invalid name: {name}")
                
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
                events = self.engine.generate_response(beat_phase=0, harmonic_context=context, bpm=120)
                self.assertEqual(events, [], f"Failed for context type: {type(context)}")

if __name__ == '__main__':
    unittest.main() 