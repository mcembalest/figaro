import unittest
from unittest.mock import MagicMock, patch
import time
import logging

from figaro import AnalysisEngine, hz_to_midi, AudioInputProcessor, Thresh

class MockInputProcessor:
    def __init__(self):
        self.onset_detector = MagicMock()
        self.onset_detector.threshold = 0.1 # Default mock threshold
        self.get_pitch = MagicMock(return_value=0.0)
        self.get_amplitude = MagicMock(return_value=0.0)

class TestAnalysisEngine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Configure logging once for this test class."""
        cls.logger = logging.getLogger('figaro')
        cls.logger.setLevel(logging.WARNING)
        if not cls.logger.hasHandlers():
            handler = logging.StreamHandler()
            cls.logger.addHandler(handler)
        cls.logger.propagate = False

    def setUp(self):
        """Set up test fixtures, including mock objects."""
        self.mock_input_processor = MagicMock(spec=AudioInputProcessor)
        self.mock_input_processor.onset_detector = MagicMock(spec=Thresh)
        self.mock_input_processor.onset_detector.threshold = 0.1
        self.mock_input_processor.get_pitch.return_value = 440.0
        self.mock_input_processor.get_amplitude.return_value = 0.5

        self.default_fs = 44100
        self.analysis_engine = AnalysisEngine(input_processor=self.mock_input_processor)

        # Reset harmonic context before each test
        self.analysis_engine.harmonic_context = None

    def tearDown(self):
        """Tear down test fixtures."""
        # Ensure AnalysisEngine resources are released if necessary
        # (Currently, stop() does nothing, but good practice)
        self.analysis_engine.stop()

    def test_process_onset_valid_pitch_and_amp(self):
        """
        Importance: Medium (Conceptually).
        Quality: Fundamentally Flawed Mocking.

        Tests the core idea: valid onset -> update context. BUT, it uses mocked
        input that provides *static, perfect* pitch/amp values. Real Pyo Yin/Follower
        data fluctuates! It doesn't test if the engine correctly handles slightly off-pitch
        notes, or amplitude values that haven't quite settled when Thresh triggers.
        Ignores the *timing* relationship between onset and feature extraction.
        Useless for proving real-world robustness.
        """
        test_time = time.time()
        test_pitch_hz = 440.0 # A4
        test_amp = 0.5
        expected_midi = hz_to_midi(test_pitch_hz) # 69
        
        # Configure mock return values
        self.mock_input_processor.get_pitch.return_value = test_pitch_hz
        self.mock_input_processor.get_amplitude.return_value = test_amp
        # Ensure mock threshold is below test_amp
        self.mock_input_processor.onset_detector.threshold = 0.1
        
        self.analysis_engine.process_onset(test_time)
        
        # Verify context updated
        self.assertEqual(self.analysis_engine.get_harmonic_context(), expected_midi)
        # Verify mocks were called
        self.mock_input_processor.get_pitch.assert_called_once()
        self.mock_input_processor.get_amplitude.assert_called_once()

    def test_process_onset_pitch_too_low(self):
        """Test onset processing when pitch is below MIN_YIN_FREQ_FOR_MONO.
        Importance: Low.
        Quality: Obvious Filter Check.

        Checks if it ignores pitches below a hardcoded threshold. Fine.
        Again, uses perfect mock data. Doesn't test edge cases around the threshold
        or how this interacts with fluctuating pitch from Yin. Minimal value.
        """
        # Need to access the class variable correctly
        min_yin_freq = AnalysisEngine.MIN_YIN_FREQ_FOR_MONO 
        test_time = time.time()
        test_pitch_hz = min_yin_freq - 10 # e.g., 20 Hz
        test_amp = 0.5
        
        # Configure mock return values
        self.mock_input_processor.get_pitch.return_value = test_pitch_hz
        self.mock_input_processor.get_amplitude.return_value = test_amp
        self.mock_input_processor.onset_detector.threshold = 0.1
        
        # Set an initial context to check it doesn't change
        initial_context = 60
        self.analysis_engine.harmonic_context = initial_context
        
        self.analysis_engine.process_onset(test_time)
        
        # Verify context remains unchanged
        self.assertEqual(self.analysis_engine.get_harmonic_context(), initial_context)

    def test_process_onset_amp_too_low(self):
        """Test onset processing when amplitude is below threshold.
        Importance: Low.
        Quality: Obvious Filter Check.

        Checks if it ignores onsets where the mocked amplitude is below the mocked threshold.
        Important filter? Yes. Tested realistically? No. It assumes amplitude is static
        and perfectly known at the onset moment. Real amplitude envelopes are dynamic.
        Ignores the complexities of Pyo's Thresh and Follower interaction.
        """
        test_time = time.time()
        test_pitch_hz = 440.0
        threshold = 0.1
        test_amp = threshold - 0.05 # Below threshold
        
        # Configure mock return values
        self.mock_input_processor.get_pitch.return_value = test_pitch_hz
        self.mock_input_processor.get_amplitude.return_value = test_amp
        self.mock_input_processor.onset_detector.threshold = threshold

        # Set an initial context
        initial_context = 72
        self.analysis_engine.harmonic_context = initial_context

        self.analysis_engine.process_onset(test_time)
        
        # Verify context remains unchanged
        self.assertEqual(self.analysis_engine.get_harmonic_context(), initial_context)

    @patch('figaro.hz_to_midi') # Patch hz_to_midi in the figaro module
    def test_process_onset_midi_conversion_fails(self, mock_hz_to_midi):
        """Test onset processing when hz_to_midi returns None.
        Importance: Low.
        Quality: Mocking Internal Helpers.
        Torvalds: "Tests what happens if 'hz_to_midi' (which is already tested in test_utils) returns None. You're patching your own utility function to test defensiveness? Pointless. Focus on testing the interaction with Pyo, not internal error handling for scenarios that the utility function tests already cover."
        """
        test_time = time.time()
        test_pitch_hz = 440.0 # A valid pitch
        test_amp = 0.5
        
        # Configure hz_to_midi mock to fail
        mock_hz_to_midi.return_value = None
        
        # Configure input processor mocks
        self.mock_input_processor.get_pitch.return_value = test_pitch_hz
        self.mock_input_processor.get_amplitude.return_value = test_amp
        self.mock_input_processor.onset_detector.threshold = 0.1

        # Set an initial context
        initial_context = 60
        self.analysis_engine.harmonic_context = initial_context

        self.analysis_engine.process_onset(test_time)
        
        # Verify context remains unchanged
        self.assertEqual(self.analysis_engine.get_harmonic_context(), initial_context)
        # Verify hz_to_midi was called with the correct pitch
        mock_hz_to_midi.assert_called_once_with(test_pitch_hz)

    def test_process_onset_context_changes(self):
        """Test a sequence of onsets causing context changes.
        Importance: Medium.
        Quality: Scripted Mock Sequence.
        Torvalds: "Runs a pre-scripted sequence of mock inputs (valid, invalid amp, valid, same valid, invalid pitch, valid different). Checks if the context state machine follows the script. Okay, this tests the intended logic in isolation. But it STILL uses perfect, static mock data for each step. It tells you NOTHING about how this logic behaves with noisy, fluctuating, slightly delayed Pyo signals in a real-time loop. The core problem remains."
        """
        base_time = time.time()
        threshold = 0.1
        self.mock_input_processor.onset_detector.threshold = threshold

        # 1. Initial valid onset (C4)
        self.mock_input_processor.get_pitch.return_value = 261.63 # C4
        self.mock_input_processor.get_amplitude.return_value = threshold + 0.1
        self.analysis_engine.process_onset(base_time + 0.1)
        self.assertEqual(self.analysis_engine.get_harmonic_context(), 60)

        # 2. Invalid onset (low amplitude)
        self.mock_input_processor.get_pitch.return_value = 440.0 # A4
        self.mock_input_processor.get_amplitude.return_value = threshold - 0.01
        self.analysis_engine.process_onset(base_time + 0.2)
        self.assertEqual(self.analysis_engine.get_harmonic_context(), 60) # Should remain C4

        # 3. Valid onset (A4)
        self.mock_input_processor.get_pitch.return_value = 440.0 # A4
        self.mock_input_processor.get_amplitude.return_value = threshold + 0.1
        self.analysis_engine.process_onset(base_time + 0.3)
        self.assertEqual(self.analysis_engine.get_harmonic_context(), 69) # Should change to A4

        # 4. Valid onset (A4 again) - Context should not change log message, but state remains
        self.mock_input_processor.get_pitch.return_value = 440.0 # A4
        self.mock_input_processor.get_amplitude.return_value = threshold + 0.2
        # Optional logging check removed for simplicity
        self.analysis_engine.process_onset(base_time + 0.4)
        self.assertEqual(self.analysis_engine.get_harmonic_context(), 69) # Should remain A4

        # 5. Invalid onset (low pitch)
        min_yin_freq = AnalysisEngine.MIN_YIN_FREQ_FOR_MONO
        self.mock_input_processor.get_pitch.return_value = min_yin_freq - 5 
        self.mock_input_processor.get_amplitude.return_value = threshold + 0.1
        self.analysis_engine.process_onset(base_time + 0.5)
        self.assertEqual(self.analysis_engine.get_harmonic_context(), 69) # Should remain A4
        
        # 6. Valid onset (G4) - Context should change back
        self.mock_input_processor.get_pitch.return_value = 392.00 # G4
        self.mock_input_processor.get_amplitude.return_value = threshold + 0.1
        self.analysis_engine.process_onset(base_time + 0.6)
        self.assertEqual(self.analysis_engine.get_harmonic_context(), 67) # Should change to G4

    def test_stop_method(self):
        """Test the stop method (currently does nothing but should exist).
        Importance: Negligible.
        Quality: Testing Nothing.
        Torvalds: "Checks if calling a currently empty 'stop' method doesn't crash. Completely pointless. If 'stop' actually did something involving Pyo resources, that would need testing. As it is, this is testing a pass statement. Delete it."
        """
        try:
            self.analysis_engine.stop()
        except Exception as e:
            self.fail(f"AnalysisEngine.stop() raised an exception: {e}")

if __name__ == '__main__':
    unittest.main() 