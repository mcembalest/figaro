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
    CHECK_INTERVAL_S,
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
@patch('figaro.Server') # Mock the Pyo Server class - NOTE: This overrides PyoServerContext behavior for Figaro instances
class TestFigaroIntegration(unittest.TestCase):
    """
    Integration tests for Figaro focusing on component interactions.
    NOTE: Due to the class-level patch of `figaro.Server`, these tests primarily
    verify interactions between mocked components, not behavior within a live
    (even null-audio) Pyo server environment. The `PyoServerContext` is defined
    but not directly utilized by the `Figaro` instances under test here.
    """

    @patch('figaro.GenerativeEngine')
    @patch('figaro.SoundEngine')
    @patch('figaro.AudioInputProcessor')
    @patch('figaro.AnalysisEngine')
    @patch('figaro.Pattern') # Patch for Pattern used in MasterScheduler
    def test_component_initialization_with_server(self, MockPattern, MockAnalysisEngine,
                                                    MockAudioInputProcessor, MockSoundEngine,
                                                    MockGenerativeEngine, MockServer):
        """
        Importance: Low (Verifies Constructor Calls).
        Quality: Heavily Mocked.

        This test mocks almost all core components and the Pyo server.
        It primarily verifies that Figaro's `__init__` calls the mocked constructors
        with expected arguments. It does not test the actual integration or behavior
        of these components within a Pyo audio processing context. Limited value for
        verifying real-time audio behavior.
        """
        mock_server_instance = MockServer
        mock_server_instance.getIsBooted.return_value = True
        mock_server_instance.getSamplingRate.return_value = 44100

        mock_audio_processor_instance = MockAudioInputProcessor.return_value
        mock_sound_engine_instance = MockSoundEngine.return_value
        mock_analysis_engine_instance = MockAnalysisEngine.return_value
        mock_generative_engine_instance = MockGenerativeEngine.return_value

        try:
            # Instantiate Figaro with the mocked server
            figaro_instance = Figaro(server=mock_server_instance)

            # Assert that component constructors were called correctly
            mock_server_instance.getIsBooted.assert_called()
            mock_server_instance.boot.assert_not_called() # Should already be booted
            MockAudioInputProcessor.assert_called_once()
            MockSoundEngine.assert_called_once()
            MockAnalysisEngine.assert_called_once_with(input_processor=mock_audio_processor_instance)
            MockGenerativeEngine.assert_called_once()
            # Assert Pattern was called (inside MasterScheduler init)
            MockPattern.assert_called_once()

            # Assert that the Figaro instance stored the components
            self.assertIs(figaro_instance.server, mock_server_instance)
            self.assertIs(figaro_instance.input_processor, mock_audio_processor_instance)
            self.assertIs(figaro_instance.sound_engine, mock_sound_engine_instance)
            self.assertIs(figaro_instance.analysis_engine, mock_analysis_engine_instance)
            self.assertIs(figaro_instance.generative_engine, mock_generative_engine_instance)

            # Assert that the MasterScheduler was implicitly created
            # Check its type (it should be the real MasterScheduler)
            self.assertIsInstance(figaro_instance.master_scheduler, MasterScheduler)
            # Check the scheduler's pattern attribute is the mocked Pattern instance
            self.assertIs(figaro_instance.master_scheduler.check_pattern, MockPattern.return_value)

        except Exception as e:
            # Capture the full traceback for debugging
            import traceback
            tb_str = traceback.format_exc()
            self.fail(f"Figaro initialization failed with mocks: {e}\nTraceback:\n{tb_str}")

    @patch('figaro.AudioInputProcessor') # Mock dependencies used by Figaro init
    @patch('figaro.SoundEngine')
    @patch('figaro.AnalysisEngine')
    @patch('figaro.GenerativeEngine')
    @patch('figaro.MasterScheduler') # Mock MasterScheduler itself
    def test_figaro_instantiates_scheduler(self, MockMasterScheduler, MockGenerativeEngine, MockAnalysisEngine, MockSoundEngine, MockAudioInputProcessor, MockServer):
        """
        Importance: Low (Verifies Constructor Calls).
        Quality: Heavily Mocked.

        Similar to the previous test, this verifies that Figaro's `__init__` calls
        the `MasterScheduler` constructor as expected. It heavily relies on mocks
        and doesn't test the scheduler's functional behavior or its interaction
        with Pyo's timing mechanisms.
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
    @patch('figaro.Pattern') # Patch for Pattern used in MasterScheduler
    def test_onset_detection_triggers_analysis(self, MockPattern, MockGenerativeEngine, MockAnalysisEngine, MockAudioInputProcessor, mock_sound_engine, mock_time, MockServer):
        """
        Importance: Medium (Concept Verification).
        Quality: Limited by Mocks.

        This test aims to verify the conceptual flow: onset detection should trigger analysis.
        However, it mocks the `Server`, `AudioInputProcessor`, and `AnalysisEngine`.
        Instead of relying on Pyo's `Thresh` to trigger the callback, it calls
        `on_onset_detected` manually. It also mocks `AnalysisEngine.process_onset`.
        Therefore, it verifies the basic debounce logic and the direct call chain,
        but not the integration with Pyo's signal processing or the analysis engine's
        internal processing of audio data.
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

    @patch('figaro.Pattern') # Patch for Pattern used in MasterScheduler
    @patch('figaro.time') # Mock time used by MasterScheduler
    @patch('figaro.AudioInputProcessor') # Mock dependencies for Figaro init
    @patch('figaro.SoundEngine')
    @patch('figaro.AnalysisEngine')
    @patch('figaro.GenerativeEngine')
    def test_scheduler_triggers_sound_on_context_change(self, MockGenerativeEngine, MockAnalysisEngine, MockSoundEngine, MockAudioInputProcessor, mock_time, MockPattern, MockServer):
        """
        Importance: High (Core Logic Flow).
        Quality: Mocked Dependencies, Tests Scheduler Logic.

        This test focuses on a critical interaction: Does a detected context change
        (simulated via mocks) lead to the `MasterScheduler` triggering the
        `GenerativeEngine` and then the `SoundEngine`?
        It mocks the core engines (`AnalysisEngine`, `SoundEngine`, `GenerativeEngine`)
        and the `Pattern` object to isolate and verify the `MasterScheduler`'s logic
        for handling context changes and initiating sound playback events.
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

        # --- Tick 1: Initial None context ---
        scheduler_callback()
        mock_sound_instance.play_harmony.assert_not_called()
        self.assertIsNone(scheduler_instance._last_played_context)

        # Simulate context changing in the AnalysisEngine mock
        new_context_note = 60 # MIDI C4
        mock_analysis_instance.get_harmonic_context.return_value = new_context_note
        current_sim_time += 1.0
        mock_time.time.return_value = current_sim_time

        # --- Tick 2: Context Change - Manually call scheduler callback again ---
        print("\n[Integration Test] Manually calling scheduler callback (tick 2 - Note context)...")

        # --- Fix: Set generate_response return value BEFORE calling the callback ---
        mock_generated_sound = MagicMock(name="GeneratedSound") # This mock isn't actually used by play_harmony, notes are
        # Correct: Return a list of event dictionaries
        # Based on GenerativeEngine logic, context 60 should yield MIDI note 60.
        expected_notes = [60]
        mock_gen_instance.generate_response.return_value = [
            {'synth': 'pad', 'action': 'play_harmony', 'midi_notes': expected_notes}
        ]
        # --- End fix ---


        # --- End add ---

        # --- Debug Prints Before Tick 2 ---
        print(f"\n[Test Debug] Before Tick 2 Callback:")
        print(f"  - Current Context Mock: {mock_analysis_instance.get_harmonic_context()}")
        print(f"  - Last Played Context Scheduler State: {scheduler_instance._last_played_context}")
        print(f"  - mock_gen_instance.generate_response call count: {mock_gen_instance.generate_response.call_count}")
        print(f"  - mock_sound_instance.play_harmony call count: {mock_sound_instance.play_harmony.call_count}")
        # --- End Debug Prints ---

        scheduler_callback()

        # --- Debug Prints After Tick 2 ---
        print(f"[Test Debug] After Tick 2 Callback:")
        print(f"  - mock_gen_instance.generate_response call count: {mock_gen_instance.generate_response.call_count}")
        if mock_gen_instance.generate_response.call_count > 0:
            print(f"  - mock_gen_instance.generate_response call args: {mock_gen_instance.generate_response.call_args}")
            print(f"  - mock_gen_instance.generate_response return value used: {mock_generated_sound}")
        print(f"  - mock_sound_instance.play_harmony call count: {mock_sound_instance.play_harmony.call_count}")
        if mock_sound_instance.play_harmony.call_count > 0:
            print(f"  - mock_sound_instance.play_harmony call args: {mock_sound_instance.play_harmony.call_args}")
        # --- End Debug Prints ---

        print("[Integration Test] Tick complete.")
        # --------------------------------------------------------------

        # Assertions
        # Verify GenerativeEngine was called, checking the actual argument order
        # Observed order seems to be: beat_phase, context, bpm
        mock_gen_instance.generate_response.assert_called_once()

        # Verify SoundEngine was called with the sound_type ('pad') and the *result* of generate_response
        # Correction: The second argument should be the midi_notes list from the generated event dictionary.
        mock_sound_instance.play_harmony.assert_called_once_with('pad', expected_notes)

        # Verify MasterScheduler's state
        self.assertEqual(scheduler_instance._last_played_context, new_context_note)
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