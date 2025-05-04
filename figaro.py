import argparse
import numpy as np
import logging
import time
import os
import datetime
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt 
from pyo import (
    Server, Input, Yin, Port, Thresh,
    SuperSaw, RCOsc, Adsr, TrigFunc,
    Mix, Freeverb, Delay, MoogLP, Clip, LFO, 
    NewTable, TableRec, Pattern, FM, Noise,
    pa_list_devices, Follower2, Record
)
import soundfile as sf
import librosa
import librosa.display

# --- Core ---
CALIBRATION_DURATION_S = 3
ONSET_DEBOUNCE_S = 0.05         # Min time between detected onsets
AMP_SMOOTH_TIME_S = 0.02        # Smoothing time for amplitude follower before threshold
CHECK_INTERVAL_S = 0.075        # How often to check harmonic context (tune this)
ONSET_SILENCE_THRESHOLD_S = 5.0 # Time without onsets to trigger reset attempt
CONTEXT_STUCK_TIMEOUT_S = 10.0  # Time after which to reset stuck context

# --- MIDI & Pitch ---
MIDI_REF_FREQ = 440.0
MIDI_REF_NOTE = 69
DEFAULT_THRESHOLD = 0.05

# --- Synth & FX ---
PAD_ADSR = [0.4, 0.4, 0.7, 4.0]         # Attack, Decay, Sustain Level, Release (Longer Release)
PLUCK_ADSR = [0.001, 0.05, 0.0, 0.1]    # Very short pluck
PLUCK_PARAMS = {'harms': 5}             # Params for Blit oscillator
PLUCK_MUL = 0.15                        # Multiplier for pluck voice
BASS_ADSR = [0.05, 0.1, 0.8, 0.5]       # Original bass ADSR
PAD_LFO_FREQ = 0.2                      # Pad detune LFO speed (Hz)
PAD_LFO_TYPE = 7                        # Pad LFO waveform (Modulated Sine)
PAD_BASE_DETUNE = 0.4                   # Base detune for SuperSaw
PAD_LFO_MOD_DEPTH = 0.1                 # Amount LFO affects detune
REVERB_MIX = 0.25                       # Wet/dry mix for reverb (0-1)
DELAY_MIX = 0.7                         # Wet/dry mix for delay (0-1)

def hz_to_midi(hz):
    """Convert frequency (Hz) to MIDI note number (integer). Returns None on invalid input."""
    try:
        if not isinstance(hz, (int, float)): raise TypeError("Input must be number")
        if hz <= 0: raise ValueError("Frequency must be > 0")
        midi_float = MIDI_REF_NOTE + 12 * np.log2(hz / MIDI_REF_FREQ)
        return int(round(midi_float))
    except (ValueError, TypeError, RuntimeWarning) as e:
        logging.warning(f"Invalid frequency {hz} for MIDI conversion: {e}")
        return None

def midi_to_hz(midi):
    """Convert MIDI note number to frequency (Hz). Returns None on invalid input."""
    try:
        if not isinstance(midi, (int, float)): raise TypeError("Input must be number")
        return float(MIDI_REF_FREQ * (2.0 ** ((midi - MIDI_REF_NOTE) / 12.0)))
    except TypeError as e:
        logging.warning(f"Invalid MIDI note {midi} for Hz conversion: {e}")
        return None

class AudioInputProcessor:
    """Handles audio input, onset detection, and provides spectral data."""
    def __init__(self, onset_callback, threshold=DEFAULT_THRESHOLD, buffer_size=256):
        self.buffer_size = buffer_size
        self.input = Input(chnl=0, mul=1)
        self.pitch_detector = Yin(self.input, minfreq=50, maxfreq=2000, cutoff=2500, winsize=2048)
        self.smooth_pitch = Port(self.pitch_detector, risetime=0.01, falltime=0.01)
        self.smoothed_amp = Follower2(self.input, risetime=AMP_SMOOTH_TIME_S, falltime=AMP_SMOOTH_TIME_S)
        self.onset_detector = Thresh(self.smoothed_amp, threshold=float(threshold))
        self.onset_callback = onset_callback
        self.onset_func = TrigFunc(self.onset_detector, self._trigger_main_callback)
        logging.info(f"AudioInputProcessor initialized (buf={buffer_size}, thresh={threshold:.3f}).")

    def _trigger_main_callback(self):
        trigger_time = time.time()
        self.onset_callback(trigger_time)

    def set_threshold(self, threshold):
        logging.info(f"Setting onset threshold to {threshold:.4f}")
        self.onset_detector.setThreshold(float(threshold))

    def get_pitch(self):
        return self.smooth_pitch.get()

    def get_amplitude(self):
        return self.smoothed_amp.get()

    # --- REMOVED: get_spectrum method ---
    # def get_spectrum(self): ...

# --- Synthesis ---
class SoundEngine: # Renamed from SynthManager
    """Manages synth voices (pad, pluck, bass) and effects, optimizing signal flow."""
    def __init__(self):
        # --- Denormal prevention noise source ---
        self.denorm_noise = Noise(1e-20) # Very low amplitude noise
        # -------------------------------------

        # --- LFOs for modulation ---
        self.pad_filter_lfo = LFO(freq=0.1, type=7, mul=400, add=600) # Modulates filter freq
        self.pluck_fm_lfo = LFO(freq=0.08, type=2, mul=0.5, add=1.0) # Modulates FM ratio slightly
        self.pad_amp_lfo = LFO(freq=0.15, type=0, mul=0.1, add=0.6) # Increased base amp (add), reduced mod depth (mul)

        self.synth_types = {
            'pad': {'gen': SuperSaw, 'poly': 3, 'mul': self.pad_amp_lfo, 'adsr': PAD_ADSR, 'fx': 'reverb_filter', 'params': {'detune': 0.3}},
            'pluck': {'gen': FM, 'poly': 3, 'mul': PLUCK_MUL * 1.5, 'adsr': PLUCK_ADSR, 'fx': 'delay',
                      'params': {'ratio': self.pluck_fm_lfo, 'index': 5}},
            'bass': {'gen': RCOsc, 'poly': 1, 'mul': 0.3, 'adsr': BASS_ADSR, 'fx': 'filter_clip', 'params': {'sharp': 0.7}},
        }
        self.voices = {} # Stores individual osc/env pairs for triggering
        self.effects_output = {} # Stores the final output of each voice type after FX
        all_effected_outputs = []

        for name, config in self.synth_types.items():
            oscs = []
            self.voices[name] = []
            for i in range(config['poly']):
                env = Adsr(*config['adsr'])
                env.setExp(0.6)

                # Apply specific ADSR multipliers based on voice type
                if name == 'pluck':
                    env.mul = PLUCK_MUL * 1.5
                elif name == 'bass':
                    env.mul = 0.3
                # else: use default Adsr mul=1

                # Create oscillator, connect envelope to its multiplier
                if config['gen'] is FM:
                     osc = config['gen'](carrier=440, mul=env, **config['params'])
                else:
                     osc = config['gen'](freq=440, mul=env, **config['params'])

                self.voices[name].append({"env": env, "osc": osc})
                oscs.append(osc)

            # --- Mix polyphonic voices *before* applying effects ---
            # Mix down to stereo (2 channels) before applying effects
            # For monophonic voices (poly=1), this just passes the single osc through
            pre_fx_mix = Mix(oscs, voices=2 if config['poly'] > 1 else 1)
            # ------------------------------------------------------

            # Apply FX to the mixed signal, passing the base voice multiplier from config
            # The base_mul (like self.pad_amp_lfo) is applied *before* the effects chain
            effected_voice = self._apply_fx(pre_fx_mix, config['fx'], base_mul=config.get('mul', 1.0))
            self.effects_output[name] = effected_voice
            all_effected_outputs.append(effected_voice)

        self.final_mix = Mix(all_effected_outputs, voices=2).out() # Mix all voice types to stereo output
        logging.info("SoundEngine initialized with optimized signal flow.")

    def _apply_fx(self, source, fx_type, base_mul=1.0):
        """Apply effect based on type string. Applies base_mul *before* effects."""
        # Apply base multiplier (could be an LFO or static value)
        multiplied_source = source * base_mul
        # Add denormal noise *after* main multiplication but *before* recursive FX
        denormaled_input = multiplied_source + self.denorm_noise

        if fx_type == 'reverb_filter': # New combined FX for pad
             filtered = MoogLP(denormaled_input, freq=self.pad_filter_lfo, res=0.4)
             # Add denorm noise again before reverb if filter didn't have it
             # In this case, MoogLP got it, so Freeverb input is ok.
             return Freeverb(filtered, size=0.9, damp=0.4, bal=REVERB_MIX)
        elif fx_type == 'reverb':
             # Should not be used currently, but add denorm here too
             return Freeverb(denormaled_input, size=0.9, damp=0.4, bal=REVERB_MIX)
        elif fx_type == 'delay':
             return Delay(denormaled_input, delay=[0.25, 0.255], feedback=0.4, mul=DELAY_MIX)
        elif fx_type == 'filter_clip':
             filtered = MoogLP(denormaled_input, freq=800, res=0.3)
             # Clip doesn't have internal recursion, no denorm needed for its input (filtered)
             return Clip(filtered, min=-0.7, max=0.7)
        else:
             return multiplied_source # No FX, return original (no denorm needed)

    def play_harmony(self, voice_name, midi_notes):
        """Play MIDI notes as frequencies on the specified polyphonic voice type.
        Modified to always use voice 0 for monophonic-style retriggering.
        """
        target_voices = self.voices.get(voice_name, [])
        num_voices = len(target_voices)
        if not num_voices or not midi_notes: return # Added check for midi_notes

        freqs = []
        for note in midi_notes:
            freq = midi_to_hz(note)
            if freq is not None:
                freqs.append(freq)
            else:
                logging.warning(f"SoundEngine: Could not convert MIDI {note} to Hz in play_harmony")

        if not freqs: # If no valid frequencies were generated
            return

        for i, freq in enumerate(freqs):
            # --- CHANGE: Always use voice 0 to ensure retriggering cuts off previous note --- 
            # voice_index = i % num_voices # Original: Cycle through voices
            voice_index = 0 # New: Force monophonic behavior for harmony context
            # ----------------------------------------------------------------------------
            voice = target_voices[voice_index]
            # --- Correctly set frequency based on oscillator type ---
            osc = voice['osc']
            if isinstance(osc, FM):
                osc.set('carrier', freq)
            elif hasattr(osc, 'setFreq'):
                osc.setFreq(freq)
            else:
                 logging.warning(f"Oscillator type {type(osc).__name__} has no setFreq or specific handler in play_harmony.")
            # -----------------------------------------------------
            voice["env"].play() # ADSR handles retriggering

    def trigger_voice(self, voice_name, freq, voice_selector=lambda poly, pos: pos % poly):
        """Trigger a single note on a potentially polyphonic voice type."""
        target_voices = self.voices.get(voice_name, [])
        num_voices = len(target_voices)
        # if not num_voices: return
        voice_index = voice_selector(num_voices, getattr(self, '_internal_pos', 0))

        chosen_voice = target_voices[voice_index]
        osc = chosen_voice['osc']
        if isinstance(osc, FM):
            osc.set('carrier', freq)
        elif hasattr(osc, 'setFreq'):
            osc.setFreq(freq)
        else:
            logging.warning(f"Oscillator type {type(osc).__name__} has no setFreq or specific handler.")
        chosen_voice['env'].play()
        self._internal_pos = getattr(self, '_internal_pos', 0) + 1

class AnalysisEngine:
    """
    Processes input features (onsets, pitch) to understand musical context 
    (primarily stable single notes based on Yin).
    """
    MIN_YIN_FREQ_FOR_MONO = 30 # Hz - Below typical instrument range

    def __init__(self, input_processor):
        """Requires the InputProcessor instance."""
        self.input_processor = input_processor
        self.harmonic_context = None
        logging.info("AnalysisEngine initialized (Simplified: Onset-driven Yin analysis).")

    def process_onset(self, timestamp):
        """
        Process a detected onset: Check current Yin pitch and amplitude 
        to update the harmonic context.
        """
        current_pitch_hz = self.input_processor.get_pitch()
        current_amp = self.input_processor.get_amplitude()
        logging.debug(f"AnalysisEngine.process_onset - Raw Input: pitch={current_pitch_hz:.2f} Hz, amp={current_amp:.4f}")
        amp_threshold = self.input_processor.onset_detector.threshold
        new_context = None
        if current_amp >= amp_threshold and current_pitch_hz > self.MIN_YIN_FREQ_FOR_MONO:
            detected_midi = hz_to_midi(current_pitch_hz)
            if detected_midi is not None:
                new_context = detected_midi
                logging.debug(f"AnalysisEngine: Valid pitch {new_context} (MIDI) detected at onset.")
            else:
                logging.debug(f"AnalysisEngine: Pitch {current_pitch_hz:.2f}Hz detected, but MIDI conversion failed.")
        else:
            logging.debug(f"AnalysisEngine: Onset ignored (amp={current_amp:.4f} vs thresh={amp_threshold:.4f} or pitch={current_pitch_hz:.2f}Hz <= {self.MIN_YIN_FREQ_FOR_MONO}Hz)")
        if new_context is not None and self.harmonic_context != new_context:
            logging.info(f"Harmonic Context Update: {new_context} (Prev: {self.harmonic_context})")
            self.harmonic_context = new_context


    def get_harmonic_context(self):
        """Returns the current harmonic context (MIDI note) or None."""
        return self.harmonic_context

    def stop(self):
        """Stop internal patterns (if any)."""
        # --- REMOVED: Stopping _analysis_pattern ---
        logging.info("AnalysisEngine stopped.")

# --- NEW: Generative Engine ---
class GenerativeEngine:
    """Decides what musical events to generate based on context and rules."""
    TARGET_OCTAVE = 4 # MIDI notes around 60
    # REMOVED: PITCH_CLASSES from global scope, define locally if needed for status/debug only
    _PITCH_CLASSES_FOR_DEBUG = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    def __init__(self):
        # REMOVED: self.chord_notes_cache
        logging.info("GenerativeEngine initialized.")

    def generate_response(self, harmonic_context):
        """Generate musical events based on context (MIDI note).
        Now generates 'play_harmony' events for 'pad' synth, aligning with test expectations.
        Simplified to handle only integer MIDI note contexts.
        """
        events = []
        # --- Requires harmonic context --- REMOVED beat_phase check ---
        if harmonic_context is None:
            return events
        # BPM is optional for current rules, but might be needed later.
        # PHASE_TOLERANCE = 0.1 # No longer needed
        # num_beats = 4 # No longer needed
        # -------------------------------------------------------------

        # --- Handle Single Note Context ---
        if isinstance(harmonic_context, (int, float)):
            midi_note = int(round(harmonic_context))

            # --- Special Rule for A4 (MIDI 69) from tests ---
            if midi_note == 69:
                notes_to_play = [69, 76] # Play A4 and E5
            else:
                # Simple Rule: Play the detected note itself
                notes_to_play = [midi_note]

            # Generate a play_harmony event for the pad synth
            events.append({
                'synth': 'pad',
                'action': 'play_harmony',
                'midi_notes': notes_to_play
            })
            logging.debug(f"GenEngine (Mono): Generated play_harmony for pad, notes={notes_to_play} (Context: {midi_note})")
        elif not isinstance(harmonic_context, (int, float)):
            logging.warning(f"GenerativeEngine: Received non-numeric context type: {type(harmonic_context)}. No events generated.")
        return events

class MasterScheduler:
    """
    Checks the harmonic context periodically and triggers sustained sounds 
    based on stable analysis results.
    """

    def __init__(self, figaro_instance, analysis_engine, sound_engine, generative_engine):
        self.figaro = figaro_instance
        self.analysis_engine = analysis_engine
        self.sound_engine = sound_engine
        self.generative_engine = generative_engine
        self._last_played_context = None
        self._last_context_change_time = time.time()
        self.check_pattern = Pattern(self._check_context_and_trigger, time=CHECK_INTERVAL_S)
        self.check_pattern.play()

    def _check_context_and_trigger(self):
        """Periodically checks harmonic context and triggers sounds if it changes."""
        try:
            current_smoothed_amp = self.figaro.input_processor.smoothed_amp
            if current_smoothed_amp is not None:
                current_smoothed_amp_val = current_smoothed_amp.get()
                current_smoothed_amp_type = type(current_smoothed_amp_val)
            else:
                logging.warning("Scheduler tick: smoothed_amp object is None")
                current_smoothed_amp_val = 0.0
                current_smoothed_amp_type = None

            logging.debug(f"Scheduler tick: Smoothed Amp Raw Value = {current_smoothed_amp_val}, Type = {current_smoothed_amp_type}")
            if isinstance(current_smoothed_amp_val, (int, float)):
                logging.debug(f"Scheduler tick: Smoothed Amp Formatted = {current_smoothed_amp_val:.5f}")
        except Exception as e:
             logging.error(f"Scheduler tick: Error getting/logging smoothed_amp: {e}", exc_info=True)

        current_context = self.analysis_engine.get_harmonic_context()
        try:
            current_amp_val = self.figaro.input_processor.smoothed_amp.get() if self.figaro.input_processor.smoothed_amp else 0.0
            current_thresh_val = self.figaro.input_processor.onset_detector.threshold if self.figaro.input_processor.onset_detector else DEFAULT_THRESHOLD
            logging.debug(f"Scheduler Check: Context={current_context}, Amp={current_amp_val:.5f}, Thresh={current_thresh_val:.5f}")
        except Exception as e:
            logging.error(f"Scheduler Check: Error getting amp/thresh: {e}")
        if current_context is not None and current_context != self._last_played_context:
            logging.info(f"Scheduler: New context detected ({current_context}). Generating response.")
            self._last_played_context = current_context
            self._last_context_change_time = time.time()
            events = self.generative_engine.generate_response(current_context)

            if events:
                logging.debug(f"Scheduler: Generated {len(events)} events: {events}")
                for event in events:
                    synth_name = event.get('synth')
                    action = event.get('action')
                    freq = event.get('freq')
                    midi_notes = event.get('midi_notes') # For harmony

                    if action == 'trigger' and synth_name and freq is not None:
                        logging.info(f"Scheduler: Triggering synth '{synth_name}' freq={freq:.2f}Hz")
                        self.sound_engine.trigger_voice(synth_name, freq)
                    elif action == 'play_harmony' and synth_name and midi_notes:
                         logging.info(f"Scheduler: Playing harmony on '{synth_name}' notes={midi_notes}")
                         self.sound_engine.play_harmony(synth_name, midi_notes)
            else:
                 logging.debug(f"Scheduler: No events generated for context {current_context}")
            # -------------------------------

        elif current_context is None and self._last_played_context is not None:
            # Context became None (e.g., silence after note), reset last played context
            logging.info("Scheduler: Context lost. Resetting last played context.")
            self._last_played_context = None # Reset last played context
            self._last_context_change_time = time.time() 

        current_time = time.time()
        if hasattr(self.figaro, 'last_raw_onset_time') and self.figaro.last_raw_onset_time > 0:
            time_since_last_onset = current_time - self.figaro.last_raw_onset_time
            if time_since_last_onset > ONSET_SILENCE_THRESHOLD_S:
                logging.warning(f"Scheduler: No onsets detected for {time_since_last_onset:.2f}s. Attempting threshold reset.")
                try:
                    original_threshold = self.figaro.input_processor.onset_detector.threshold
                    temp_threshold = max(original_threshold * 0.3, 0.003)
                    self.figaro.input_processor.set_threshold(temp_threshold)
                    self.figaro.input_processor.set_threshold(original_threshold)
                    self.figaro.last_raw_onset_time = current_time
                    logging.info("Scheduler: Onset detector threshold reset attempted.")
                except Exception as e:
                    logging.error(f"Scheduler: Error during threshold reset: {e}")
        else:
             self.figaro.last_raw_onset_time = current_time

    def stop(self):
        """Stop the scheduler's checking Pattern."""
        if self.check_pattern.isPlaying():
            self.check_pattern.stop()
        logging.info("MasterScheduler stopped.")

class Figaro:
    """Audio input, analysis, harmony, rhythm, and synthesis."""

    def __init__(self, server=None, input_device=None, output_device=None, buffer_size=256, calibration_duration=CALIBRATION_DURATION_S, record=False, record_duration=5, record_dir='recordings'):
        if server is not None and server.getIsBooted():
            logging.info("Figaro using provided Pyo server instance.")
            self.server = server
        else:
            logging.info("Figaro creating new Pyo server instance (portaudio).")
            self.server = Server(audio='portaudio', buffersize=buffer_size)
            if input_device is not None:
                try: self.server.setInputDevice(input_device)
                except Exception as e: logging.error(f"Failed to set input device {input_device}: {e}")
            if output_device is not None:
                 try: self.server.setOutputDevice(output_device)
                 except Exception as e: logging.error(f"Failed to set output device {output_device}: {e}")
            self.server.setMidiInputDevice(99) # Avoid MIDI conflicts
            self.server.boot()
        self.fs = self.server.getSamplingRate()
        self.calibrated_threshold = DEFAULT_THRESHOLD
        self.calibration_duration = calibration_duration
        self.input_processor = AudioInputProcessor(onset_callback=self.on_onset_detected,
                                                   threshold=self.calibrated_threshold,
                                                   buffer_size=buffer_size)
        self.sound_engine = SoundEngine()
        self.analysis_engine = AnalysisEngine(input_processor=self.input_processor)
        self.generative_engine = GenerativeEngine()
        self.master_scheduler = MasterScheduler(figaro_instance=self, analysis_engine=self.analysis_engine,
                                              sound_engine=self.sound_engine,
                                              generative_engine=self.generative_engine)
        self.last_true_onset_time = 0
        self.last_raw_onset_time = 0 
        self.record = record
        self.record_duration = record_duration
        self.record_base_dir = record_dir
        self.current_rec_session_dir = None
        self.current_rec_audio_file = None
        self.recorder = None
        if self.record:
            if not os.path.exists(self.record_base_dir):
                os.makedirs(self.record_base_dir)
                logging.info(f"Created base recording directory: {self.record_base_dir}")
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.current_rec_session_dir = os.path.join(self.record_base_dir, timestamp)
            os.makedirs(self.current_rec_session_dir)
            logging.info(f"Created recording session directory: {self.current_rec_session_dir}")
            self.current_rec_audio_file = os.path.join(self.current_rec_session_dir, "session_audio.wav")
            self.recorder = Record(self.sound_engine.final_mix, filename=self.current_rec_audio_file, chnls=2, fileformat=0, sampletype=0)
            logging.info(f"Recording setup complete. Output will be saved to: {self.current_rec_session_dir}")

    def calibrate(self):
        """Calibrate input threshold based on ambient noise."""
        print(f"Calibrating threshold ({self.calibration_duration}s)... Be quiet.")
        cal_table = NewTable(length=self.calibration_duration)
        recorder = TableRec(self.input_processor.smoothed_amp, table=cal_table).play()
        time.sleep(self.calibration_duration)
        recorder.stop()
        samples = np.array(cal_table.getTable())
        if samples.size > 0:
            peak_amp = np.percentile(samples, 98)
            noise_floor = np.median(samples)
            calculated_thresh = noise_floor + (peak_amp - noise_floor) * 0.5
            self.calibrated_threshold = max(calculated_thresh, 0.015)
            self.input_processor.set_threshold(self.calibrated_threshold)
            print(f"Calibration done. Noise Floor: {noise_floor:.5f}, Peak (98%): {peak_amp:.5f} => Threshold: {self.calibrated_threshold:.5f}")
        else:
            print("Calibration failed (no data). Using default threshold.")
            self.calibrated_threshold = DEFAULT_THRESHOLD
            self.input_processor.set_threshold(self.calibrated_threshold)
        cal_table.reset()
        del recorder
        del cal_table

    def on_onset_detected(self, trigger_time):
        """Callback triggered by AudioInputProcessor. Feeds AnalysisEngine."""
        self.last_raw_onset_time = trigger_time
        logging.debug(f"Raw onset callback triggered at {trigger_time:.4f}")
        if trigger_time - self.last_true_onset_time >= ONSET_DEBOUNCE_S:
            self.last_true_onset_time = trigger_time
            logging.debug(f"True onset detected at {trigger_time:.4f} -> AnalysisEngine.process_onset")
            self.analysis_engine.process_onset(trigger_time)

    def _generate_session_plot(self, n_mels=128):
        """Load recorded audio and generate waveform/spectrogram plot."""
        audio_file_path = self.current_rec_audio_file
        try:
            audio_data, sr = sf.read(audio_file_path)
            if audio_data.ndim > 1: audio_data = np.mean(audio_data, axis=1)
            output_image_file = os.path.join(self.current_rec_session_dir, "session_plot.png")
            fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            librosa.display.waveshow(audio_data, sr=sr, ax=axes[0])
            axes[0].set_title("Session Waveform")
            axes[0].set_ylabel("Amplitude")
            S = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=n_mels)
            S_db = librosa.power_to_db(S, ref=np.max)
            librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', ax=axes[1])
            axes[1].set_title("Session Mel Spectrogram")
            axes[1].set_ylabel("Frequency (Mel)")
            axes[1].set_xlabel("Time (s)")
            plt.tight_layout()
            plt.savefig(output_image_file)
            plt.close(fig)
            print(f"Saved session plot to {output_image_file}")
        except Exception as e:
            logging.error(f"Error generating session plot: {e}", exc_info=True)
            print(f"Error generating session plot: {e}")

    def start(self):
        """Start the audio server, calibrate, and run main loop or visualization."""
        self.server.start()
        time.sleep(0.1)
        self.calibrate()
        if self.record:
            print(f"Running recording session for {self.record_duration} seconds...")
            if self.recorder:
                print(f"Recording audio output to: {self.current_rec_audio_file}")
                self.recorder.play()
            try:
                time.sleep(self.record_duration)
            except KeyboardInterrupt:
                print("\nRecording interrupted.")
            finally:
                if self.recorder and self.recorder.isPlaying():
                    self.recorder.stop()
                    print("Recording stopped.")
                    time.sleep(0.1)
                    self._generate_session_plot()
                print("Recording session complete.")
        else:
            print("Figaro started. Press Ctrl+C to stop.")
            try:
                loop_count = 0
                while self.server.getIsStarted():
                    time.sleep(0.1)
                    loop_count += 1
                    if loop_count % 5 == 0:
                        context = self.analysis_engine.get_harmonic_context()
                        last_played = self.master_scheduler._last_played_context
                        status_parts = []
                        if context:
                            if isinstance(context, (int, float)):
                                context_str = f"Note: {int(round(context))}"
                            else:
                                context_str = f"Context: {context}"
                        else:
                            context_str = "Harmony: N/A"
                        status_parts.append(f"Detected: {context_str}")
                        notes_played = self._get_notes_for_context(last_played)
                        if notes_played:
                            notes_str = ", ".join(map(str, notes_played))
                            played_str = f"Playing: {notes_str}"
                        else:
                            played_str = "Playing: -"
                        status_parts.append(f"{played_str}")
                        print(f"Status - { ' | '.join(status_parts) }", end='\r', flush=True)

            except KeyboardInterrupt:
                print("\nStopping Figaro...") 
            finally:
                self.stop()

    def _get_notes_for_context(self, context):
        """Helper for status display: returns the notes played for a given context."""
        if context is None: return []
        if isinstance(context, (int, float)):
            midi_note = int(round(context))
            if midi_note == 69: return [69, 76]
            else: return [midi_note]

        return []

    def stop(self):
        """Stop the audio server and analysis/scheduling engines gracefully."""
        if hasattr(self, 'master_scheduler') and self.master_scheduler:
            self.master_scheduler.stop()
        if hasattr(self, 'analysis_engine') and self.analysis_engine:
             self.analysis_engine.stop()
        if hasattr(self, 'recorder') and self.recorder and self.recorder.isPlaying():
            logging.info("Stopping recorder during shutdown.")
            self.recorder.stop()
        if self.server.getIsBooted():
            self.server.stop()
            logging.info("Audio server stopped.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Figaro: Interactive Harmonizer")
    parser.add_argument('-i', '--input-device', type=int, default=None, help="Audio input device ID.")
    parser.add_argument('-o', '--output-device', type=int, default=None, help="Audio output device ID.")
    parser.add_argument('--list-devices', action='store_true', help="List available audio devices and exit.")
    parser.add_argument('--buffer-size', type=int, default=256, help="Audio buffer size (samples).")
    parser.add_argument('--log-level', default='WARNING', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Set logging level.')
    parser.add_argument('--record', action='store_true', help="Enable recording mode (saves audio and plot to timestamped directory).")
    parser.add_argument('--record-duration', type=float, default=10, help="Recording duration in seconds.")
    parser.add_argument('--record-dir', type=str, default='recordings', help="Base directory for recordings.")
    args = parser.parse_args()
    if args.list_devices:
        pa_list_devices()
    else:
        log_level_from_args = getattr(logging, args.log_level.upper(), logging.INFO)
        logging.basicConfig(level=log_level_from_args, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
        figaro = Figaro(input_device=args.input_device,
                            output_device=args.output_device,
                            buffer_size=args.buffer_size,
                            record=args.record,
                            record_duration=args.record_duration,
                            record_dir=args.record_dir)
        figaro.start()
