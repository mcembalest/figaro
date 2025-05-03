import unittest
import time
from pyo import Server, Sig
import logging
import os
import unittest.mock
import math
from unittest.mock import patch, MagicMock, ANY
import numpy as np
import sys

from figaro import (
    Figaro,
    AudioInputProcessor,
    AnalysisEngine,
    SoundEngine,
    GenerativeEngine,
    MasterScheduler,
    # ContextAnalyzer, # Removed/Renamed
    # HarmonyGenerator, # Removed/Renamed
    # BeatPredictorScheduler, # Removed/Renamed
    hz_to_midi, # Import for mocking/testing
    DEFAULT_THRESHOLD,
    ONSET_DEBOUNCE_S,
    CONTEXT_STUCK_TIMEOUT_S,
)

# --- Global Test Configuration ---
# Set to True if you have a loopback audio device configured for testing
# (e.g., BlackHole, Soundflower, JACK, VB-Cable) and know its input/output device IDs.
USE_REAL_AUDIO_LOOPBACK = os.environ.get('FIGARO_TEST_AUDIO', 'false').lower() == 'true'
# Replace with your actual loopback device IDs if USE_REAL_AUDIO_LOOPBACK is True
LOOPBACK_INPUT_DEVICE_ID = None  # e.g., 2
LOOPBACK_OUTPUT_DEVICE_ID = None # e.g., 3

# --- Pyo Server Context Manager (inspired by pyo tests) ---
class PyoServerContext:
    "Context manager to start/stop the Pyo server for tests."
    def __init__(self, use_null_audio=True, input_dev=None, output_dev=None, buffer_size=256):
        # Ensure null audio is used unless explicitly overridden *and* testing loopback
        self.use_null_audio = True # Default to True for safety in tests
        if not use_null_audio and USE_REAL_AUDIO_LOOPBACK:
            logging.warning("PyoServerContext configured for REAL audio loopback.")
            self.use_null_audio = False
            
        self.input_dev = input_dev if input_dev is not None else (LOOPBACK_INPUT_DEVICE_ID if not self.use_null_audio else 99)
        self.output_dev = output_dev if output_dev is not None else (LOOPBACK_OUTPUT_DEVICE_ID if not self.use_null_audio else 99) # Use 99 for null output too
        self.buffer_size = buffer_size
        self.server = None
        self.sr = 44100 # Assume standard SR, adjust if needed
        self.bs = buffer_size
        self._is_owner = False # Flag to track if this context started the server

    def __enter__(self):
        logging.info(f"Entering PyoServerContext (null_audio={self.use_null_audio})...")
        audio_driver = "null" if self.use_null_audio else "portaudio"
        
        # Check if server is already running - avoid creating multiple servers
        global_server = Server(audio=audio_driver) # Create temporary Server obj to check state
        if global_server.getIsBooted():
            logging.warning("PyoServerContext: Server already booted. Reusing existing server.")
            self.server = global_server
            self._is_owner = False # This context did not start it
            # Use existing server's parameters
            self.sr = self.server.getSamplingRate()
            self.bs = self.server.getBufferSize()
        else:
            # No server running, boot a new one
            logging.info(f"Attempting server boot (driver='{audio_driver}')...")
            try:
                self.server = Server(audio=audio_driver, buffersize=self.buffer_size).boot()
                if not self.server.getIsBooted():
                    raise RuntimeError("Server instance failed to boot.")
                    
                if not self.use_null_audio:
                    logging.info(f"Configuring PortAudio devices: Input={self.input_dev}, Output={self.output_dev}")
                    self.server.setInputDevice(self.input_dev)
                    self.server.setOutputDevice(self.output_dev)
                else:
                    # Explicitly set dummy devices for null audio if needed?
                    # Usually not necessary, but safer?
                    self.server.setInputDevice(99)
                    self.server.setOutputDevice(99)
                    
                self.server.setMidiInputDevice(99) # Avoid MIDI conflicts
                self.server.start()
                time.sleep(0.1) # Short pause for server startup
                if not self.server.getIsStarted():
                    raise RuntimeError("Server did not start correctly.")
                    
                self._is_owner = True # This context started the server
                self.sr = self.server.getSamplingRate()
                self.bs = self.server.getBufferSize()
                logging.info(f"Server started by context (SR={self.sr}, BS={self.bs}).")
                
            except Exception as e:
                logging.error(f"PyoServerContext failed to start server: {e}", exc_info=True)
                if self.server and self.server.getIsBooted():
                    # Clean up if boot started but failed later
                    if self.server.getIsStarted(): self.server.stop()
                    self.server.shutdown() 
                self.server = None
                self._is_owner = False
                raise # Re-raise exception to signal failure

        return self # Return context manager instance itself

    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.info("Exiting PyoServerContext...")
        # Only stop/shutdown the server if this context instance started it
        if self.server and self._is_owner:
            if self.server.getIsBooted():
                try:
                    if self.server.getIsStarted(): self.server.stop()
                    self.server.shutdown()
                    logging.info("Server stopped and shut down by owner context.")
                except Exception as e:
                     logging.error(f"Error stopping/shutting down server in PyoServerContext: {e}", exc_info=True)
            else:
                 logging.warning("Owner context exiting, but server was not booted?")
        elif self.server and not self._is_owner:
            logging.info("Exiting PyoServerContext that reused existing server (leaving server running).")
        else:
             logging.warning("Exiting PyoServerContext, server instance was None.")
             
        self.server = None # Clear reference
        self._is_owner = False
        # Don't suppress exceptions
        return False

    def process_time(self, duration_s):
        "Manually process server audio for a given duration."
        if not self.server or not self.server.getIsStarted():
             logging.warning("Server not running, cannot process time.")
             return
        num_buffers = int(math.ceil(duration_s * self.sr / self.bs))
        if num_buffers <= 0:
            return
        logging.debug(f"Processing {num_buffers} buffers ({duration_s}s)...")
        # Directly call _process instead of process to avoid potential GUI hangs in tests
        for _ in range(num_buffers):
            self.server._process() # Use internal process call
        logging.debug("Processing finished.")

@patch('figaro.Figaro.calibrate', MagicMock()) # Prevent actual calibration
@patch('figaro.Server') # Mock the Pyo Server class - KEEP THIS ONE
class TestFigaroIntegration(unittest.TestCase):
    """
    Integration tests for Figaro focusing on component interactions within the Pyo environment.
    Uses PyoServerContext with null_audio=True and manual time processing.
    """

    @patch('figaro.AudioInputProcessor')
    @patch('figaro.SoundEngine')
    @patch('figaro.AnalysisEngine')
    @patch('figaro.GenerativeEngine')
    @patch('figaro.MasterScheduler')
    def test_component_initialization_with_server(self, MockMasterScheduler, MockGenerativeEngine, MockAnalysisEngine, MockSoundEngine, MockAudioInputProcessor, MockServer):
        """
        Importance: Laughable.
        Quality: Mock Hell.

        You call this integration? You've mocked EVERY SINGLE CLASS! This tests *nothing*
        about how these components actually interact using Pyo signals or timing.
        It just checks if your __init__ calls other __init__s with mock objects.
        This is the definition of useless. Provides ZERO confidence the actual application works.
        DELETE THIS TEST. It is actively harmful by creating a false sense of security.
        """
        mock_server_instance = MockServer.return_value
        mock_server_instance.getSamplingRate.return_value = 44100
        mock_server_instance.getIsBooted.return_value = True # Simulate booted server

        # Mock constructors to return MagicMock instances
        mock_audio_processor_instance = MockAudioInputProcessor.return_value
        mock_sound_engine_instance = MockSoundEngine.return_value
        mock_analysis_engine_instance = MockAnalysisEngine.return_value
        mock_generative_engine_instance = MockGenerativeEngine.return_value
        mock_master_scheduler_instance = MockMasterScheduler.return_value

        try:
            # Initialize Figaro, using the mocked server instance
            figaro_instance = Figaro(server=mock_server_instance)

            # Assertions: Check if components were initialized
            MockAudioInputProcessor.assert_called_once()
            # Check args passed to AudioInputProcessor (might need refinement)
            self.assertEqual(MockAudioInputProcessor.call_args[1]['onset_callback'], figaro_instance.on_onset_detected)

            MockSoundEngine.assert_called_once()
            MockAnalysisEngine.assert_called_once_with(input_processor=mock_audio_processor_instance, fs=44100)
            MockGenerativeEngine.assert_called_once()
            MockMasterScheduler.assert_called_once_with(
                figaro_instance=figaro_instance,
                analysis_engine=mock_analysis_engine_instance,
                sound_engine=mock_sound_engine_instance,
                generative_engine=mock_generative_engine_instance
            )

            # Check that attributes are set on the instance
            self.assertIsInstance(figaro_instance.input_processor, MagicMock)
            self.assertIsInstance(figaro_instance.sound_engine, MagicMock)
            self.assertIsInstance(figaro_instance.analysis_engine, MagicMock)
            self.assertIsInstance(figaro_instance.generative_engine, MagicMock)
            self.assertIsInstance(figaro_instance.master_scheduler, MagicMock)

        except Exception as e:
            self.fail(f"Figaro initialization failed with mocks: {e}")

    @patch('figaro.AudioInputProcessor') # Mock dependencies used by Figaro init
    @patch('figaro.SoundEngine')
    @patch('figaro.AnalysisEngine')
    @patch('figaro.GenerativeEngine')
    @patch('figaro.MasterScheduler') # Mock MasterScheduler itself
    def test_figaro_instantiates_scheduler(self, MockMasterScheduler, MockGenerativeEngine, MockAnalysisEngine, MockSoundEngine, MockAudioInputProcessor, MockServer):
        """
        Importance: Laughable.
        Quality: More Mock Hell.

        Another monumentally pointless test. It checks if Figaro's __init__ calls
        MasterScheduler's __init__? Are you testing if Python can execute code sequentially?
        This tells us NOTHING about whether the scheduler actually *schedules* anything
        or interacts correctly with Pyo's timing. ZERO INTEGRATION VALUE.
        Delete this garbage too.
        """
        mock_server_instance = MockServer.return_value
        mock_server_instance.getSamplingRate.return_value = 44100
        mock_server_instance.getIsBooted.return_value = True

        # Get mock instances that will be created/used
        mock_analysis_instance = MockAnalysisEngine.return_value
        mock_sound_instance = MockSoundEngine.return_value
        mock_gen_instance = MockGenerativeEngine.return_value

        # Instantiate Figaro
        figaro_instance = Figaro(server=mock_server_instance)

        # Assert MasterScheduler was called once with the correct arguments
        MockMasterScheduler.assert_called_once_with(
            figaro_instance=figaro_instance,
            analysis_engine=mock_analysis_instance,
            sound_engine=mock_sound_instance,
            generative_engine=mock_gen_instance
        )
        # Assert the instance attribute is set
        self.assertIsInstance(figaro_instance.master_scheduler, MagicMock)

    @patch('figaro.time') # Keep this inner patch
    @patch('figaro.SoundEngine') # Add patch for SoundEngine
    @patch('figaro.AudioInputProcessor') # Also patch AudioInputProcessor
    @patch('figaro.AnalysisEngine') # And AnalysisEngine
    @patch('figaro.GenerativeEngine') # And GenerativeEngine
    @patch('figaro.Pattern') # <<< ADD THIS MOCK
    def test_onset_detection_triggers_analysis(self, MockPattern, MockGenerativeEngine, MockAnalysisEngine, MockAudioInputProcessor, mock_sound_engine, mock_time, MockServer):
        """
        Importance: High (Conceptually).
        Quality: Utterly Neutered by Mocks.

        Okay, the *idea* is important: onset -> analysis. But you mock EVERYTHING AGAIN!
        You manually call `on_onset_detected`? You mock `AnalysisEngine`?
        This test proves *nothing* about whether a real Pyo `Thresh` object triggering
        `on_onset_detected` actually causes the *real* `AnalysisEngine` to process
        the *real*, potentially noisy, pitch/amplitude data from Pyo at the right time.
        This mock-fest completely sidesteps all the real-time challenges.
        Needs a complete rewrite to use *real* (offline) Pyo objects.
        """
        mock_server_instance = MockServer.return_value
        mock_server_instance.getSamplingRate.return_value = 44100
        # --- CRITICAL FIX: Ensure server reports as booted BEFORE Figaro is created ---
        mock_server_instance.getIsBooted.return_value = True
        # --- END FIX ---

        # Mock dependencies needed *within* Figaro.__init__
        mock_audio_processor_instance = MockAudioInputProcessor.return_value
        mock_analysis_instance = MockAnalysisEngine.return_value

        # Use real Figaro, but with mocked dependencies
        figaro_instance = Figaro(server=mock_server_instance)

        # Now, mock the specific method *on the Figaro instance's attribute*
        figaro_instance.analysis_engine.process_onset = MagicMock()

        # Simulate onsets
        test_time_1 = 100.0
        test_time_2 = 100.03 # Too soon (debounce)
        test_time_3 = 100.10 # OK

        # --- Simulate callback directly ---
        # First onset
        figaro_instance.on_onset_detected(test_time_1)
        figaro_instance.analysis_engine.process_onset.assert_called_once_with(test_time_1)
        self.assertEqual(figaro_instance.last_true_onset_time, test_time_1)

        # Second (debounced) onset
        figaro_instance.on_onset_detected(test_time_2)
        # Should still be called only once
        figaro_instance.analysis_engine.process_onset.assert_called_once_with(test_time_1)
        self.assertEqual(figaro_instance.last_true_onset_time, test_time_1) # Last true onset time shouldn't update

        # Third (valid) onset
        figaro_instance.on_onset_detected(test_time_3)
        # Now called twice
        self.assertEqual(figaro_instance.analysis_engine.process_onset.call_count, 2)
        figaro_instance.analysis_engine.process_onset.assert_called_with(test_time_3) # Check last call
        self.assertEqual(figaro_instance.last_true_onset_time, test_time_3)

    @patch('figaro.Pattern') # Mock Pattern used by MasterScheduler
    @patch('figaro.time') # Mock time used by MasterScheduler
    @patch('figaro.AudioInputProcessor') # Mock dependencies for Figaro init
    @patch('figaro.SoundEngine')
    @patch('figaro.AnalysisEngine')
    @patch('figaro.GenerativeEngine')
    def test_scheduler_triggers_sound_on_context_change(self, MockGenerativeEngine, MockAnalysisEngine, MockSoundEngine, MockAudioInputProcessor, mock_time, MockPattern, MockServer):
        """
        Importance: Critical.
        Quality: Completely Faked "Integration".

        This is supposed to be the CORE loop: analysis context -> scheduler -> sound.
        And what do you do? Mock the Scheduler's Pattern, mock time, mock the AnalysisEngine,
        mock the SoundEngine! You manually set the AnalysisEngine's return value and then
        manually call the Scheduler's *internal* check method? And assert the mocked SoundEngine
        was called?
        This isn't an integration test! This is a puppet show!
        It proves absolutely NOTHING about whether the real system works with Pyo.
        It ignores timing, signal flow, potential race conditions, everything.
        This needs a complete rewrite using an offline Pyo server and *actual* Pyo signals,
        feeding simulated input and checking the *actual* output signal or state changes.
        This current version is worse than useless.
        """
        mock_server_instance = MockServer.return_value
        mock_server_instance.getSamplingRate.return_value = 44100
        mock_server_instance.getIsBooted.return_value = True

        # Mock instances created during Figaro init
        mock_analysis_instance = MockAnalysisEngine.return_value
        mock_sound_instance = MockSoundEngine.return_value
        mock_gen_instance = MockGenerativeEngine.return_value

        # Mock time
        current_sim_time = 100.0
        mock_time.time.return_value = current_sim_time

        # Mock AnalysisEngine behavior
        mock_analysis_instance.get_harmonic_context.return_value = None # Initial state

        # Instantiate Figaro (which implicitly creates MasterScheduler)
        figaro_instance = Figaro(server=mock_server_instance)

        # Get the actual MasterScheduler instance created
        # We need to get the callback function the real scheduler passed to Pattern
        scheduler_instance = figaro_instance.master_scheduler
        self.assertIsInstance(scheduler_instance, MasterScheduler) # Ensure it's the real one

        # Find the actual callback function passed to Pattern
        self.assertTrue(MockPattern.called)
        scheduler_callback = MockPattern.call_args[0][0] # Get the function argument
        self.assertEqual(scheduler_callback.__name__, '_check_context_and_trigger')

        # --- Simulate scheduler tick 1: No context ---
        scheduler_callback()
        mock_sound_instance.play_harmony.assert_not_called()
        self.assertIsNone(scheduler_instance._last_played_context)

        # --- Simulate context change: Note detected ---
        new_context_note = 60 # MIDI C4
        mock_analysis_instance.get_harmonic_context.return_value = new_context_note
        current_sim_time += 1.0
        mock_time.time.return_value = current_sim_time

        # --- Simulate scheduler tick 2: Context changed ---
        scheduler_callback()
        # Expect pad voice, single note list
        mock_sound_instance.play_harmony.assert_called_once_with('pad', [new_context_note])
        self.assertEqual(scheduler_instance._last_played_context, new_context_note)
        self.assertEqual(scheduler_instance._last_context_change_time, current_sim_time)

        # --- Simulate scheduler tick 3: Context stable ---
        current_sim_time += 1.0
        mock_time.time.return_value = current_sim_time
        scheduler_callback()
        # Should not be called again
        mock_sound_instance.play_harmony.assert_called_once() # Still called only once

        # --- Simulate context change: Chord detected ---
        new_context_chord = 'C_maj'
        expected_notes = [60, 64, 67] # Assuming GenerativeEngine works correctly
        mock_analysis_instance.get_harmonic_context.return_value = new_context_chord
        # Mock the generative engine helper needed by scheduler's check
        mock_gen_instance._get_chord_midi_notes.return_value = expected_notes
        current_sim_time += 1.0
        mock_time.time.return_value = current_sim_time

        # --- Simulate scheduler tick 4: Context changed to chord ---
        scheduler_callback()
        # Should be called twice now
        self.assertEqual(mock_sound_instance.play_harmony.call_count, 2)
        mock_sound_instance.play_harmony.assert_called_with('pad', expected_notes) # Check last call
        self.assertEqual(scheduler_instance._last_played_context, new_context_chord)
        self.assertEqual(scheduler_instance._last_context_change_time, current_sim_time)

# --- Test Suite Execution ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
    print("Running Figaro Integration Tests...")
    unittest.main()

# --- Removed Helper Functions (midi_to_hz) as they are not needed directly ---
# --- Removed USE_REAL_AUDIO_LOOPBACK logic as no tests use it currently ---

# Helper function from figaro.py (copied for test_context_to_harmony_flow)
MIDI_REF_FREQ = 440.0
MIDI_REF_NOTE = 69
HARMONY_CONFIDENCE_THRESHOLD = 0.3 # Define locally if not imported

def midi_to_hz(midi):
    if not isinstance(midi, (int, float)):
        raise ValueError("MIDI note must be a number.")
    return float(MIDI_REF_FREQ * (2.0 ** ((midi - MIDI_REF_NOTE) / 12.0))) 