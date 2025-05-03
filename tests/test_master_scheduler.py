import unittest
from unittest.mock import MagicMock, patch, call, ANY
import logging

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
    ONSET_SILENCE_THRESHOLD_S
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

    def setUp(self):
        """Set up mocks for Figaro components and time/Pattern."""
        # Create mocks directly within setUp
        self.mock_pattern_class = MagicMock()
        self.mock_time_module = MagicMock()
        self.mock_pattern_instance = self.mock_pattern_class.return_value
        self.mock_time_func = self.mock_time_module.time
        
        self.current_time = 100.0
        self.mock_time_func.return_value = self.current_time

        # Mocks for Figaro and components
        self.mock_figaro = MagicMock(spec=Figaro) 
        self.mock_analysis_engine = MagicMock(spec=AnalysisEngine)
        self.mock_sound_engine = MagicMock(spec=SoundEngine)
        self.mock_generative_engine = MagicMock(spec=GenerativeEngine)
        self.mock_input_processor = MagicMock(spec=AudioInputProcessor)

        self.mock_figaro.analysis_engine = self.mock_analysis_engine
        self.mock_figaro.sound_engine = self.mock_sound_engine
        self.mock_figaro.generative_engine = self.mock_generative_engine
        self.mock_figaro.input_processor = self.mock_input_processor

        self.mock_input_processor.onset_detector = MagicMock()
        self.mock_input_processor.onset_detector.threshold = 0.1
        self.mock_input_processor.set_threshold = MagicMock()

        self.mock_figaro.last_raw_onset_time = 0
        self.mock_analysis_engine.get_harmonic_context.return_value = None
        
        # Instantiate the scheduler with mocks
        # Since pyo.Server is already booted in setUpClass,
        # we don't need to patch Pattern anymore - it will use the real pyo.Pattern
        # but we still need to patch time as that's used in the application logic
        with patch('figaro.time', self.mock_time_module):
             self.scheduler = MasterScheduler(
                 figaro_instance=self.mock_figaro, 
                 analysis_engine=self.mock_analysis_engine, 
                 sound_engine=self.mock_sound_engine,
                 generative_engine=self.mock_generative_engine
             )
             # Get a reference to the real Pattern instance for inspection
             self.pattern_instance = self.scheduler.check_pattern


    def advance_time(self, seconds):
        """Helper to advance mock time."""
        self.current_time += seconds
        self.mock_time_func.return_value = self.current_time

    @patch('figaro.time')
    def test_initialization(self, mock_time):
        """
        Importance: Pathetic.
        Quality: Garbage.

        Checks if attributes are initialized? Are you serious? A glorified type checker.
        Does it verify the core scheduling Pattern is ACTUALLY running, or configured
        with the *correct* interval? Hell no. Utterly useless for testing if this thing
        will even *try* to react in real-time. Fix it or delete it.
        """
        # With real Pattern, we can only verify the state, not the play() call
        # Since we use real pyo.Pattern, we can't use assert_called_once on it
        self.assertEqual(self.scheduler._last_played_context, None)
        self.assertEqual(self.scheduler._last_context_change_time, self.current_time)

    @patch('figaro.time')
    def test_context_change_none_to_note(self, mock_time):
        """
        Importance: High.
        Quality: Delusional Mockery.

        Okay, you *claim* to test the core reaction: note detected -> play pad.
        But you mock EVERYTHING. This doesn't prove squat about Pyo interaction,
        timing, or reliability. Does it check if play_harmony is called *promptly*?
        Does it simulate context jitter? No. It's a fantasy, not a test.
        Make it interact with a *real* (offline) Pyo process or admit defeat.
        """
        # Set mock time return value via the decorator's mock
        mock_time.time.return_value = self.current_time
        
        new_context_note = 60 # C4
        expected_notes = get_expected_notes_for_context(new_context_note)
        self.mock_analysis_engine.get_harmonic_context.return_value = new_context_note
        
        # Call the method under test
        self.scheduler._check_context_and_trigger()
        
        self.mock_sound_engine.play_harmony.assert_called_once_with('pad', expected_notes)
        self.assertEqual(self.scheduler._last_played_context, new_context_note)
        self.assertEqual(self.scheduler._last_context_change_time, self.current_time)

    @patch('figaro.time')
    def test_context_change_note_to_chord(self, mock_time):
        """
        Importance: High.
        Quality: More Mocking Nonsense.

        Same pathetic story as the note test, just with a different imaginary context.
        Still avoids any *real* testing of Pyo integration, timing, or edge cases.
        Does it check if the *old* sound stops correctly before the new one starts?
        Of course not. You're just checking if your mocks talk to each other. Worthless.
        """
        mock_time.time.return_value = self.current_time
        initial_context_note = 60
        self.scheduler._last_played_context = initial_context_note
        self.mock_analysis_engine.get_harmonic_context.return_value = initial_context_note 
        last_change_time = self.current_time - 1.0
        self.scheduler._last_context_change_time = last_change_time
        
        new_context_chord = 'G_maj'
        expected_notes = get_expected_notes_for_context(new_context_chord)
        self.mock_analysis_engine.get_harmonic_context.return_value = new_context_chord
        
        # --- Add this line to configure the mock ---
        self.mock_generative_engine._get_chord_midi_notes.return_value = expected_notes
        # -------------------------------------------

        self.scheduler._check_context_and_trigger()
        
        self.mock_sound_engine.play_harmony.assert_called_once_with('pad', expected_notes)
        self.assertEqual(self.scheduler._last_played_context, new_context_chord)
        self.assertEqual(self.scheduler._last_context_change_time, self.current_time)

    @patch('figaro.time')
    def test_context_change_chord_to_none(self, mock_time):
        """
        Importance: Medium.
        Quality: Trivial.

        Checks if it *stops* calling play_harmony when context goes None. Whoop-dee-doo.
        Does it verify the *sound* actually decays gracefully in Pyo? Nope.
        Just checks that your mocked sound engine *isn't* called. Minimal value.
        Ensuring silence works is less critical than ensuring sound *starts* correctly.
        """
        mock_time.time.return_value = self.current_time
        initial_context_chord = 'G_maj'
        self.scheduler._last_played_context = initial_context_chord
        self.mock_analysis_engine.get_harmonic_context.return_value = initial_context_chord
        last_change_time = self.current_time - 1.0
        self.scheduler._last_context_change_time = last_change_time

        self.mock_analysis_engine.get_harmonic_context.return_value = None
        
        self.scheduler._check_context_and_trigger()
        
        self.mock_sound_engine.play_harmony.assert_not_called()
        self.assertEqual(self.scheduler._last_played_context, None)
        self.assertEqual(self.scheduler._last_context_change_time, self.current_time)

    @patch('figaro.time')
    def test_context_stable_no_trigger(self, mock_time):
        """
        Importance: Low.
        Quality: Obvious.

        Tests that if nothing changes, nothing happens. Groundbreaking stuff.
        Again, purely mock-based, tells us nothing about Pyo state or real-world
        performance. Did the sound *sustain* correctly during this period? Test doesn't care.
        Barely worth the bytes it occupies.
        """
        mock_time.time.return_value = self.current_time
        initial_context_note = 60
        self.scheduler._last_played_context = initial_context_note
        self.mock_analysis_engine.get_harmonic_context.return_value = initial_context_note
        last_change_time = self.current_time - 1.0
        self.scheduler._last_context_change_time = last_change_time

        self.scheduler._check_context_and_trigger()
        
        self.mock_sound_engine.play_harmony.assert_not_called()
        self.assertEqual(self.scheduler._last_played_context, initial_context_note)
        self.assertEqual(self.scheduler._last_context_change_time, last_change_time) # Should not update
        
    @patch('figaro.time')
    def test_special_a4_context(self, mock_time):
        """
        Importance: Low.
        Quality: Specific, but Still Mocked.

        Okay, you test one specific hardcoded rule (A4 -> A+E). Fine.
        But it's *still* just checking mock calls. Zero insight into whether Pyo
        actually plays the correct two notes, or if they sound right together.
        Testing arbitrary magic numbers is less important than testing the core engine.
        """
        mock_time.time.return_value = self.current_time
        new_context_note = 69
        expected_notes = [69, 76]
        self.mock_analysis_engine.get_harmonic_context.return_value = new_context_note
        
        self.scheduler._check_context_and_trigger()
        
        self.mock_sound_engine.play_harmony.assert_called_once_with('pad', expected_notes)
        self.assertEqual(self.scheduler._last_played_context, new_context_note)
        self.assertEqual(self.scheduler._last_context_change_time, self.current_time)

    @patch('figaro.time')
    def test_stuck_context_timeout(self, mock_time):
        """
        Importance: Medium.
        Quality: Embarrassing Time Mockery.

        Testing the stuck context reset? Necessary feature. Testing it by fast-forwarding
        mock time? Lazy and avoids the *real* issues. Does this reset happen *reliably*
        without race conditions with incoming context changes? Does it work if Pyo's event
        loop is under heavy load? This test wouldn't know. It lives in fantasy land.
        """
        initial_time = self.current_time
        mock_time.time.return_value = initial_time # Set initial time via decorator mock
        
        initial_context_note = 60
        self.scheduler._last_played_context = initial_context_note
        self.mock_analysis_engine.get_harmonic_context.return_value = initial_context_note
        initial_change_time = initial_time 
        self.scheduler._last_context_change_time = initial_change_time
        
        # Advance time just past the timeout relative to initial time
        stuck_time = initial_time + CONTEXT_STUCK_TIMEOUT_S + 0.1
        mock_time.time.return_value = stuck_time # Update mock return value for the check call
        
        self.scheduler._check_context_and_trigger()
        
        # Verify context was reset
        self.assertIsNone(self.mock_analysis_engine.harmonic_context) # Check the attribute was set to None
        self.assertEqual(self.scheduler._last_played_context, None)
        self.assertEqual(self.scheduler._last_context_change_time, stuck_time) 
        self.mock_sound_engine.play_harmony.assert_not_called()

    @patch('figaro.time')
    def test_onset_silence_reset(self, mock_time):
        """
        Importance: Medium-High.
        Quality: Incompetent Mock Verification.

        The idea (resetting threshold on silence) is crucial for adapting to live input.
        The test? Garbage. More time mocking. It checks `set_threshold` was called,
        but does it check *what values* were used? Does it ensure the dip-then-restore
        logic is correct? Does it verify `last_raw_onset_time` was actually updated
        to prevent immediate re-triggering? No. This is testing by wishful thinking.
        Useless for proving robustness.
        """
        mock_time.time.return_value = self.current_time
        self.mock_figaro.last_raw_onset_time = self.current_time - (ONSET_SILENCE_THRESHOLD_S + 1.0)
        original_threshold = 0.15
        self.mock_input_processor.onset_detector.threshold = original_threshold 
        
        self.scheduler._check_context_and_trigger()
        
        expected_dip_threshold = max(original_threshold * 0.3, 0.003)
        expected_calls = [
            call(expected_dip_threshold),
            call(original_threshold)
        ]
        self.mock_input_processor.set_threshold.assert_has_calls(expected_calls) 
        self.assertEqual(self.mock_figaro.last_raw_onset_time, self.current_time) # Should update last onset time
        
    @patch('figaro.time')
    def test_stop_method(self, mock_time):
        """
        Importance: Low.
        Quality: Trivial Mock Check.

        Checks if calling stop() calls the pattern's stop()? Seriously?
        This is testing Python's ability to call methods. Completely pointless.
        Does it verify resources are *actually* released in Pyo? No. Delete this.
        """
        # Ensure the pattern is playing initially (setUp starts it)
        self.assertTrue(self.pattern_instance.isPlaying())
        
        # Call stop
        self.scheduler.stop()
        
        # Verify Pattern is no longer playing
        self.assertFalse(self.pattern_instance.isPlaying())
        
        # Verify no more callbacks are triggered after stop
        mock_time.time.return_value = self.current_time + 1.0
        self.scheduler._check_context_and_trigger()
        self.mock_sound_engine.play_harmony.assert_not_called()
        
        # Verify state is cleaned up
        self.assertIsNone(self.scheduler._last_played_context)


if __name__ == '__main__':
    unittest.main() 