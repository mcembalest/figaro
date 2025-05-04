import unittest
from unittest.mock import MagicMock, patch, call, ANY
import logging
import time

# Import pyo Server class before importing Figaro modules
from pyo import Server

# Assuming figaro.py is in the parent directory or accessible via PYTHONPATH
from figaro import (
    MasterScheduler, 
    GenerativeEngine, 
    AnalysisEngine,
    SoundEngine,
    AudioInputProcessor,
    Figaro,
    CONTEXT_STUCK_TIMEOUT_S,
    ONSET_SILENCE_THRESHOLD_S,
    DEFAULT_THRESHOLD
)

logging.disable(logging.CRITICAL)

# --- Helper to get expected notes (mirrors logic in Figaro/Scheduler) ---
# Necessary because we mock GenerativeEngine and SoundEngine
def get_expected_notes_for_context(context):
    if context is None: return []
    if isinstance(context, (int, float)):
        midi_note = int(round(context))
        if midi_note == 69: return [69, 76] # Special A4 case
        else: return [midi_note]
    elif isinstance(context, str):
        # Simplified mock logic for chord notes - assumes format 'X_maj' or 'X_min'
        # Uses GenerativeEngine default octave
        try:
            root_str, quality = context.split('_')
            # We need PITCH_CLASSES from figaro, import it or define locally
            PITCH_CLASSES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            root_pc = PITCH_CLASSES.index(root_str)
            # Use TARGET_OCTAVE from the actual GenerativeEngine class
            root_midi = GenerativeEngine.TARGET_OCTAVE * 12 + root_pc 
            if quality == 'maj': intervals = [0, 4, 7]
            elif quality == 'min': intervals = [0, 3, 7]
            else: intervals = [0, 4, 7] # Default assumption for tests
            return [root_midi + i for i in intervals]
        except:
            return [] # Error parsing
    return []
# ----------------------------------------------------------------------

# Mock necessary pyo objects if they are used directly or their methods are called
class MockPyoPattern:
    def __init__(self, function, time):
        self._function = function
        self._time = time
        self._playing = False

    def play(self):
        self._playing = True

    def stop(self):
        self._playing = False

    def isPlaying(self):
        return self._playing

    # --- Add a method to manually trigger the callback for testing ---
    def tick(self):
        if self._playing:
            self._function()

class MockFollower2:
    def __init__(self, *args, **kwargs):
        self._value = 0.0
    def get(self):
        return self._value
    def set_value(self, val): # Helper for tests
        self._value = val

class MockThresh:
    def __init__(self, *args, **kwargs):
        self.threshold = kwargs.get('threshold', DEFAULT_THRESHOLD)
    def setThreshold(self, val):
        self.threshold = val


class TestMasterScheduler(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up the pyo server for all tests in this class."""
        # Create a dummy/offline server that doesn't need real audio hardware
        cls.pyo_server = Server(sr=44100, buffersize=256, audio="offline", nchnls=2)
        # Boot the server - required before creating any audio objects
        cls.pyo_server.boot()

    @classmethod
    def tearDownClass(cls):
        """Tear down the pyo server after all tests in this class are done."""
        # Stop the server if it's running
        if cls.pyo_server.getIsStarted():
            cls.pyo_server.stop()
        # Shut down the server
        cls.pyo_server.shutdown()

    @patch('figaro.Pattern', new=MockPyoPattern) # Mock pyo.Pattern
    def setUp(self):
        """Set up mocks for each test."""
        # Mock the dependent classes
        self.mock_analysis_engine = MagicMock(spec=AnalysisEngine)
        self.mock_sound_engine = MagicMock(spec=SoundEngine)
        self.mock_generative_engine = MagicMock(spec=GenerativeEngine)
        self.mock_figaro_instance = MagicMock(spec=Figaro)

        # Mock Figaro's input processor and its relevant components
        self.mock_input_processor = MagicMock(spec=AudioInputProcessor)
        self.mock_input_processor.smoothed_amp = MockFollower2()
        self.mock_input_processor.onset_detector = MockThresh()
        self.mock_figaro_instance.input_processor = self.mock_input_processor
        self.mock_figaro_instance.last_raw_onset_time = time.time() # Initialize

        # Configure mock analysis engine return values for default cases
        self.mock_analysis_engine.get_harmonic_context.return_value = None

        # --- Ensure harmonic_context attribute exists ---
        # Option 1: Make it part of the spec (if AnalysisEngine truly has it)
        # Option 2: Add it explicitly to the mock instance here
        self.mock_analysis_engine.harmonic_context = None 
        # ------------------------------------------------------------

        # Configure mock generative engine default return value
        self.mock_generative_engine.generate_response.return_value = []

        # Instantiate the MasterScheduler with mocks
        self.scheduler = MasterScheduler(
            figaro_instance=self.mock_figaro_instance,
            analysis_engine=self.mock_analysis_engine,
            sound_engine=self.mock_sound_engine,
            generative_engine=self.mock_generative_engine
        )
        # Ensure the pattern starts (as it does in __init__)
        self.scheduler.check_pattern.play()
        # Reset internal state for safety between tests
        self.scheduler._last_played_context = None
        self.scheduler._last_context_change_time = time.time()

    @patch('figaro.Pattern', new=MockPyoPattern)
    def test_initialization(self):
        """Importance: Pathetic."""
        self.assertIsNotNone(self.scheduler)
        self.assertTrue(self.scheduler.check_pattern.isPlaying())
        self.assertIsNone(self.scheduler._last_played_context)

    @patch('figaro.Pattern', new=MockPyoPattern)
    def test_context_stable_no_trigger(self):
        """Importance: Low."""
        # Set initial context
        initial_context = 60 # MIDI C4
        self.mock_analysis_engine.get_harmonic_context.return_value = initial_context
        self.scheduler._last_played_context = initial_context
        self.scheduler._last_context_change_time = time.time()

        # Mock generate_response to return a dummy event
        self.mock_generative_engine.generate_response.return_value = [{'synth': 'pad', 'action': 'play_harmony', 'midi_notes': [initial_context]}]

        # Trigger the scheduler's check multiple times
        self.scheduler.check_pattern.tick() 
        self.scheduler.check_pattern.tick() 

        # Assertions
        # generate_response should NOT have been called again after the first hypothetical time
        self.mock_generative_engine.generate_response.assert_not_called() 
        # Sound engine methods should NOT have been called
        self.mock_sound_engine.trigger_voice.assert_not_called()
        self.mock_sound_engine.play_harmony.assert_not_called()

    @patch('figaro.Pattern', new=MockPyoPattern)
    def test_context_change_none_to_note(self):
        """Importance: High."""
        self.mock_analysis_engine.get_harmonic_context.return_value = None
        self.scheduler.check_pattern.tick() # Process initial None state
        self.mock_generative_engine.generate_response.assert_not_called()
        self.mock_sound_engine.play_harmony.assert_not_called()
        self.assertIsNone(self.scheduler._last_played_context)
        new_context_note = 60 # MIDI C4
        expected_notes = [new_context_note]
        self.mock_analysis_engine.get_harmonic_context.return_value = new_context_note
        self.mock_generative_engine.generate_response.return_value = [
            {'synth': 'pad', 'action': 'play_harmony', 'midi_notes': expected_notes}
        ]
        self.scheduler.check_pattern.tick()
        self.mock_generative_engine.generate_response.assert_called_once_with(new_context_note)
        self.mock_sound_engine.play_harmony.assert_called_once_with('pad', expected_notes)
        self.assertEqual(self.scheduler._last_played_context, new_context_note)

    @patch('figaro.Pattern', new=MockPyoPattern)
    def test_context_change_note_to_chord(self):
        """Importance: High."""
        # Initial state: context is a note
        initial_note_context = 60 # C4
        initial_expected_notes = [initial_note_context]
        self.mock_analysis_engine.get_harmonic_context.return_value = initial_note_context
        self.mock_generative_engine.generate_response.return_value = [
             {'synth': 'pad', 'action': 'play_harmony', 'midi_notes': initial_expected_notes}
        ]
        self.scheduler.check_pattern.tick() # Process initial state
        self.mock_generative_engine.generate_response.assert_called_once_with(initial_note_context)
        self.mock_sound_engine.play_harmony.assert_called_once_with('pad', initial_expected_notes)
        self.assertEqual(self.scheduler._last_played_context, initial_note_context)
        # Reset mocks for the next phase
        self.mock_generative_engine.generate_response.reset_mock()
        self.mock_sound_engine.play_harmony.reset_mock()

        # Change context to a chord
        new_chord_context = "G_maj"
        expected_notes = [55, 59, 62] # G4, B4, D5 (Target Octave 4)
        self.mock_analysis_engine.get_harmonic_context.return_value = new_chord_context
        self.mock_generative_engine.generate_response.return_value = [
            {'synth': 'pad', 'action': 'play_harmony', 'midi_notes': expected_notes}
        ]

        # Trigger the scheduler's check
        self.scheduler.check_pattern.tick()

        # Assertions
        self.mock_generative_engine.generate_response.assert_called_once_with(new_chord_context)
        self.mock_sound_engine.play_harmony.assert_called_once_with('pad', expected_notes)
        self.assertEqual(self.scheduler._last_played_context, new_chord_context)

    @patch('figaro.Pattern', new=MockPyoPattern)
    def test_context_change_chord_to_none(self):
        """Importance: Medium."""
        # Initial state: context is a chord
        initial_chord_context = "G_maj"
        initial_expected_notes = [55, 59, 62]
        self.mock_analysis_engine.get_harmonic_context.return_value = initial_chord_context
        self.mock_generative_engine.generate_response.return_value = [
             {'synth': 'pad', 'action': 'play_harmony', 'midi_notes': initial_expected_notes}
        ]
        self.scheduler.check_pattern.tick() # Process initial state
        self.assertEqual(self.scheduler._last_played_context, initial_chord_context)
        # Reset mocks
        self.mock_generative_engine.generate_response.reset_mock()
        self.mock_sound_engine.play_harmony.reset_mock()

        # Change context to None
        self.mock_analysis_engine.get_harmonic_context.return_value = None
        # Trigger the scheduler's check
        self.scheduler.check_pattern.tick()

        # Assertions
        # Generative engine should NOT be called when context becomes None
        self.mock_generative_engine.generate_response.assert_not_called()
        # Sound engine should NOT be called (we let sounds decay naturally for now)
        self.mock_sound_engine.play_harmony.assert_not_called()
        # Scheduler should update its internal state to None
        self.assertEqual(self.scheduler._last_played_context, None)

    @patch('figaro.Pattern', new=MockPyoPattern)
    def test_special_a4_context(self):
        """Importance: Low."""
        # Test the specific handling for MIDI note 69 (A4)
        a4_context = 69
        expected_notes = [69, 76] # A4, E5
        self.mock_analysis_engine.get_harmonic_context.return_value = a4_context
        self.mock_generative_engine.generate_response.return_value = [
            {'synth': 'pad', 'action': 'play_harmony', 'midi_notes': expected_notes}
        ]

        # Trigger the scheduler's check
        self.scheduler.check_pattern.tick()

        # Assertions
        self.mock_generative_engine.generate_response.assert_called_once_with(a4_context)
        self.mock_sound_engine.play_harmony.assert_called_once_with('pad', expected_notes)
        self.assertEqual(self.scheduler._last_played_context, a4_context)

    @patch('figaro.Pattern', new=MockPyoPattern)
    @patch('time.time') # Mock time
    def test_onset_silence_reset(self, mock_time):
        """Importance: Medium-High."""
        initial_time = 1000.0
        mock_time.return_value = initial_time
        self.mock_figaro_instance.last_raw_onset_time = initial_time

        original_threshold = 0.05
        self.mock_input_processor.onset_detector.threshold = original_threshold

        # 1. Time hasn't advanced enough - no reset
        mock_time.return_value = initial_time + ONSET_SILENCE_THRESHOLD_S / 2
        self.scheduler.check_pattern.tick()
        self.mock_input_processor.set_threshold.assert_not_called()

        # 2. Advance time beyond the threshold
        mock_time.return_value = initial_time + ONSET_SILENCE_THRESHOLD_S + 0.1
        self.scheduler.check_pattern.tick()

        # Assertions: Threshold should have been temporarily lowered and restored
        expected_temp_threshold = max(original_threshold * 0.3, 0.003)
        # Check calls to set_threshold
        self.assertEqual(self.mock_input_processor.set_threshold.call_count, 2)
        # Check the first call was to lower it
        self.mock_input_processor.set_threshold.assert_any_call(expected_temp_threshold)
        # Check the second call was to restore it
        self.mock_input_processor.set_threshold.assert_any_call(original_threshold)
        # Check the last_raw_onset_time was updated
        self.assertEqual(self.mock_figaro_instance.last_raw_onset_time, mock_time.return_value)

    @patch('figaro.Pattern', new=MockPyoPattern)
    def test_stop_method(self):
        """Importance: Low."""
        self.assertTrue(self.scheduler.check_pattern.isPlaying())
        self.scheduler.stop()
        self.assertFalse(self.scheduler.check_pattern.isPlaying())


if __name__ == '__main__':
    unittest.main() 