import argparse
import numpy as np
import logging
import time
from collections import deque
from pyo import (
    Server, Input, Yin, Port, Thresh,
    SuperSaw, RCOsc, Adsr, TrigFunc,
    Mix, Freeverb, Delay, MoogLP, Clip, LFO, Spectrum,
    NewTable, TableRec, Pattern, FM, Noise,
    pa_list_devices, Follower2
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

# --- Core Parameters ---
CALIBRATION_DURATION_S = 3
ONSET_CUTOFF_TIME_S = 5.0       # Max age of onset times for rhythm analysis
ONSET_DEBOUNCE_S = 0.05         # Min time between detected onsets
AMP_SMOOTH_TIME_S = 0.02       # Smoothing time for amplitude follower before threshold
NOTE_HISTORY_DURATION_S = 10.0  # Max age of notes for key analysis
BEAT_CHECK_INTERVAL_S = 0.05
CONTEXT_STUCK_TIMEOUT_S = 10.0  # Time after which to reset stuck context

# --- MIDI/Pitch Related ---
MIDI_REF_FREQ = 440.0
MIDI_REF_NOTE = 69
RHYTHM_PITCH_MIDI = 72
DEFAULT_THRESHOLD = 0.05
MIN_NOTES_FOR_CONTEXT = 5
NOTE_HISTORY_LENGTH = 100 # Max notes in history for context analysis
HARMONY_CONFIDENCE_THRESHOLD = 0.3
MIN_IOI_S = 0.1
MAX_IOI_S = 2.0
IOI_HIST_BINS = 50
MIN_IOIS_FOR_ANALYSIS = 5
BEAT_SMOOTHING_ALPHA = 0.3
BEAT_CONFIDENCE_THRESHOLD = 0.4
SLOW_BPM_THRESHOLD = 80
MEDIUM_BPM_THRESHOLD = 120

# --- Synth & FX Parameters (Adjustable) ---
PAD_ADSR = [0.4, 0.4, 0.7, 4.0]       # Attack, Decay, Sustain Level, Release (Longer Release)
PLUCK_ADSR = [0.001, 0.05, 0.0, 0.1]    # Very short pluck
PLUCK_PARAMS = {'harms': 5}             # Params for Blit oscillator
PLUCK_MUL = 0.15                         # Multiplier for pluck voice
BASS_ADSR = [0.05, 0.1, 0.8, 0.5]       # Original bass ADSR
PAD_LFO_FREQ = 0.2                     # Pad detune LFO speed (Hz)
PAD_LFO_TYPE = 7                       # Pad LFO waveform (Modulated Sine)
PAD_BASE_DETUNE = 0.4                  # Base detune for SuperSaw
PAD_LFO_MOD_DEPTH = 0.1                # Amount LFO affects detune
REVERB_MIX = 0.25                       # Wet/dry mix for reverb (0-1)
DELAY_MIX = 0.7                        # Wet/dry mix for delay (0-1)

# Krumhansl Profiles (Normalized)
KRUMHANSL_MAJOR_PROFILE = np.array([
    6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.63, 2.24, 2.88
])
KRUMHANSL_MINOR_PROFILE = np.array([
    6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17
])

# --- Utility Functions ---
def hz_to_midi(hz):
    """Convert frequency (Hz) to MIDI note number (integer). Returns None on invalid input."""
    try:
        # Validate input type and value
        if not isinstance(hz, (int, float)): raise TypeError("Input must be number")
        if hz <= 0: raise ValueError("Frequency must be > 0")
        # Calculate MIDI note (float)
        midi_float = MIDI_REF_NOTE + 12 * np.log2(hz / MIDI_REF_FREQ)
        # Return rounded integer MIDI note
        return int(round(midi_float))
    except (ValueError, TypeError, RuntimeWarning) as e: # Catch log2 domain error via RuntimeWarning
        logging.warning(f"Invalid frequency {hz} for MIDI conversion: {e}")
        return None

def midi_to_hz(midi):
    """Convert MIDI note number to frequency (Hz). Returns None on invalid input."""
    try:
        # Validate input type
        if not isinstance(midi, (int, float)): raise TypeError("Input must be number")
        # Calculate frequency
        # Ensure float output for pyo
        return float(MIDI_REF_FREQ * (2.0 ** ((midi - MIDI_REF_NOTE) / 12.0)))
    except TypeError as e:
        logging.warning(f"Invalid MIDI note {midi} for Hz conversion: {e}")
        return None

# --- NEW: Chroma Feature Calculation ---
N_CHROMA_BINS = 12
N_OCTAVES = 7 # Covering MIDI 24 to 108 approx
MIDI_START_NOTE = 24 # C1
FFT_SIZE = 1024 # Needs tuning based on buffer size and desired resolution
FFT_OVERLAPS = 4 # Standard is 4 overlaps for FFT

# --- Chord Templates (Normalized Chroma Vectors) ---
# Templates derived from theoretical distributions. Can be refined.
PITCH_CLASSES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
CHORD_TEMPLATES = {}

def _create_chord_template(intervals):
    template = np.zeros(N_CHROMA_BINS)
    for interval in intervals:
        template[interval % N_CHROMA_BINS] = 1.0
    # Simple normalization (sum to 1)
    norm = np.linalg.norm(template, ord=1)
    return template / norm if norm > 0 else template

# Major Triads
for i in range(N_CHROMA_BINS):
    root_name = PITCH_CLASSES[i]
    CHORD_TEMPLATES[f'{root_name}_maj'] = np.roll(_create_chord_template([0, 4, 7]), i)
# Minor Triads
for i in range(N_CHROMA_BINS):
    root_name = PITCH_CLASSES[i]
    CHORD_TEMPLATES[f'{root_name}_min'] = np.roll(_create_chord_template([0, 3, 7]), i)
# Add more templates as needed (e.g., Dom7, Dim, Aug...)
# ----------------------------------------------------

def calculate_chroma(spectrum_data, fs, fft_size=FFT_SIZE):
    """Calculate a 12-element chroma vector from pyo Spectrum data."""
    # --- Add logging for spectrum_data structure ---
    spec_type = type(spectrum_data)
    spec_len = -1
    first_elem_type = None
    spec_stats = "N/A"
    if isinstance(spectrum_data, (list, np.ndarray)):
        spec_len = len(spectrum_data)
        if spec_len > 0:
            first_elem_type = type(spectrum_data[0])
            # --- ENHANCED DEBUG LOG ---
            try:
                spec_np = np.asarray(spectrum_data)
                if spec_np.size > 0: # Check for empty array after conversion
                    spec_stats = f"min={np.min(spec_np):.4f}, max={np.max(spec_np):.4f}, mean={np.mean(spec_np):.4f}"
                else:
                    spec_stats = "empty_array_after_conversion"
            except Exception as e:
                 spec_stats = f"error_calculating_stats: {e}"
            # --- END ENHANCED DEBUG LOG ---

    # --- Use INFO level temporarily to ensure visibility ---
    logging.info(f"calculate_chroma INPUT: Type={spec_type}, Len={spec_len}, FirstElemType={first_elem_type}, Stats=[{spec_stats}]")
    # -------------------------------------------------------

    if spectrum_data is None or spec_len <= 0: # Use checked spec_len, ensure > 0
        logging.warning(f"calculate_chroma: Returning zeros due to invalid or empty input (len={spec_len}).") # Add warning
        return np.zeros(N_CHROMA_BINS)

    # Frequency resolution of the FFT bins
    freq_resolution = fs / fft_size
    # Frequencies corresponding to each FFT bin index
    # Ensure spectrum_data is treated as the array length source
    fft_freqs = np.arange(spec_len) * freq_resolution 

    chroma_vector = np.zeros(N_CHROMA_BINS)
    total_energy = 0.0

    # Calculate frequencies for MIDI notes C1 to B7 (approx)
    midi_note_freqs = [midi_to_hz(MIDI_START_NOTE + i) for i in range(N_OCTAVES * 12)]

    # --- Simplified Mapping: Assign energy of nearest FFT bin to pitch class ---
    # More sophisticated methods exist (e.g., triangular weighting), but start simple.
    for i in range(N_OCTAVES * 12):
        target_freq = midi_note_freqs[i]
        if target_freq is None:
            continue
        
        # Find the FFT bin closest to the target frequency
        closest_bin_index = int(round(target_freq / freq_resolution))
        
        # Ensure index is within bounds of the spectrum data
        if 0 <= closest_bin_index < spec_len: # Use checked spec_len
            raw_energy_value = spectrum_data[closest_bin_index]
            # --- Add logging for raw_energy_value ---
            logging.debug(f"calculate_chroma: bin={closest_bin_index}, raw_energy_value Type={type(raw_energy_value)}, Value={raw_energy_value}")
            # -----------------------------------------

            # --- Attempt to extract scalar energy ---
            # Assuming it might be a single-element list/array/tuple, try accessing the first element
            if isinstance(raw_energy_value, (list, np.ndarray, tuple)) and len(raw_energy_value) > 0:
                energy = raw_energy_value[0] # Assume magnitude is the first element
            elif isinstance(raw_energy_value, (int, float)): # Or maybe it's already scalar sometimes?
                energy = raw_energy_value
            else:
                energy = 0 # Cannot determine energy, skip this bin
                logging.warning(f"calculate_chroma: Could not extract scalar energy from bin {closest_bin_index}. Type: {type(raw_energy_value)}")
            # ----------------------------------------
            
            # Now check the extracted scalar energy
            if np.isfinite(energy) and energy > 0:
                 # Map MIDI note index (0-83) to pitch class (0-11)
                 pitch_class = (MIDI_START_NOTE + i) % N_CHROMA_BINS
                 chroma_vector[pitch_class] += energy
                 total_energy += energy

    # Normalize the chroma vector
    if total_energy > 1e-6: # Avoid division by zero or near-zero
        chroma_vector /= total_energy
    
    return chroma_vector
# -------------------------------------

# --- NEW: Spectral Flatness Calculation ---
def calculate_spectral_flatness(spectrum_data):
    """Calculate Spectral Flatness Measure (SFM). Closer to 1 = more noise-like."""
    # --- Add type check for robustness ---
    if not isinstance(spectrum_data, (list, np.ndarray)) or len(spectrum_data) == 0:
        logging.debug("SFM Debug: Input invalid or empty, returning 1.0")
        # Return 1.0 (max flatness) if input is invalid, empty, or not sequence-like
        return 1.0
    # ------------------------------------
    
    # Use magnitude spectrum (Spectrum object provides magnitudes)
    # --- Explicitly cast to float64 to handle integer input from Spectrum callback ---
    magnitudes = np.asarray(spectrum_data, dtype=np.float64)
    # ---------------------------------------------------------------------------
    # Add small epsilon to avoid log(0) or division by zero
    epsilon = 1e-10
    magnitudes += epsilon
    
    num_bins = len(magnitudes)
    # --- SFM Debug ---
    magnitudes_min = np.min(magnitudes)
    magnitudes_max = np.max(magnitudes)
    magnitudes_mean = np.mean(magnitudes)
    logging.debug(f"SFM Debug: Input Array (len={num_bins}, min={magnitudes_min:.4e}, max={magnitudes_max:.4e}, mean={magnitudes_mean:.4e}) after adding epsilon={epsilon:.1e}")
    # -----------------
    # This check is now technically redundant due to the initial check, but safe to keep
    if num_bins == 0:
         logging.debug("SFM Debug: num_bins is 0, returning 1.0")
         return 1.0
         
    # Geometric mean
    # Filter out non-positive values before log to avoid warnings/errors
    # Note: Since we added epsilon > 0, all values should technically be positive now.
    # Let's check anyway for safety.
    positive_magnitudes = magnitudes[magnitudes > 0]
    num_positive = len(positive_magnitudes)
    # --- SFM Debug ---
    logging.debug(f"SFM Debug: Found {num_positive} positive magnitudes (out of {num_bins})")
    # -----------------

    if num_positive == 0:
        logging.debug("SFM Debug: No positive magnitudes found, returning 1.0")
        return 1.0 # If no positive magnitudes, treat as maximally flat
        
    log_sum = np.sum(np.log(positive_magnitudes))
    # Use num_positive for the geometric mean calculation if filtering occurred
    geometric_mean = np.exp(log_sum / num_positive) 
    
    # Arithmetic mean (use original magnitudes including epsilon)
    # Re-calculate arithmetic_mean based on the potentially filtered positive_magnitudes for consistency?
    # No, SFM definition typically uses the mean of *all* bins. Let's stick to that.
    arithmetic_mean = np.mean(magnitudes)
    
    # --- SFM Debug ---
    logging.debug(f"SFM Debug: GeoMean={geometric_mean:.4e}, ArithMean={arithmetic_mean:.4e}")
    # -----------------

    if arithmetic_mean < epsilon: # Check against epsilon, not just zero
         logging.debug(f"SFM Debug: Arithmetic mean ({arithmetic_mean:.4e}) < epsilon ({epsilon:.1e}), returning 1.0")
         return 1.0 # Avoid division by zero if signal is essentially zero
         
    sfm = geometric_mean / arithmetic_mean
    # --- SFM Debug ---
    logging.debug(f"SFM Debug: Calculated SFM = {sfm:.4f} (before clipping)")
    # -----------------
    # Clamp SFM to [0, 1] range for safety
    clipped_sfm = np.clip(sfm, 0.0, 1.0)
    if clipped_sfm != sfm:
        logging.debug(f"SFM Debug: SFM clipped to {clipped_sfm:.4f}")
        
    return clipped_sfm
# -----------------------------------------

# --- Key Detection ---
class KrumhanslKeyDetector:
    """Detects musical key via Krumhansl-Schmuckler algorithm."""
    def __init__(self):
        self.major_profiles = {i: np.roll(KRUMHANSL_MAJOR_PROFILE, i) for i in range(12)}
        self.minor_profiles = {i: np.roll(KRUMHANSL_MINOR_PROFILE, i) for i in range(12)}
        self.total_analyses = 0
        self.total_analysis_time = 0
        self.max_history = 500
        self.confidence_history = deque(maxlen=self.max_history)
        logging.info("KrumhanslKeyDetector initialized.")

    def analyze(self, notes):
        """Analyzes MIDI notes list for key. Returns {key, mode, confidence} or None."""
        valid_notes = self._validate_notes(notes)
        if not valid_notes or len(valid_notes) < MIN_NOTES_FOR_CONTEXT:
             return None 

        start_time = time.time()
        try:
            pitch_classes = np.array(valid_notes) % 12
            histogram = np.bincount(pitch_classes, minlength=12).astype(float)
            
            hist_sum = np.sum(histogram)
            if hist_sum == 0:
                return None 
            histogram /= hist_sum

            correlations = []
            for key in range(12):
                major_corr = self._calculate_correlation(histogram, self.major_profiles[key])
                minor_corr = self._calculate_correlation(histogram, self.minor_profiles[key])
                correlations.extend([(key, 'major', major_corr), (key, 'minor', minor_corr)])

            correlations.sort(key=lambda x: x[2], reverse=True)
            best_key, best_mode, max_correlation = correlations[0]
            
            confidence = self._calculate_confidence(histogram, max_correlation, correlations)
            self._update_profiling(time.time() - start_time, confidence)
            
            # key_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
            # logging.debug(...) 
            
            return {'key': best_key, 'mode': best_mode, 'confidence': confidence}
            
        except Exception as e:
            logging.error(f"KrumhanslKeyDetector: Unexpected error in analysis: {e}", exc_info=True)
            return None

    def _validate_notes(self, notes):
        """Validate notes list. Filters invalid notes and logs warnings."""
        if not isinstance(notes, list):
            # Still raise ValueError here, as incorrect type is a fundamental usage error.
            raise ValueError("Input 'notes' must be a list.")
        if not notes: return [] 
        
        valid_notes = []
        for i, note in enumerate(notes):
            try:
                if note is None: raise ValueError("is None")
                # Attempt conversion cautiously
                note_val = float(str(note).strip())
                if not note_val.is_integer(): raise ValueError("is not an integer")
                note_int = int(note_val)
                if not (0 <= note_int <= 127): raise ValueError("out of range [0-127]")
                valid_notes.append(note_int)
            except (ValueError, TypeError, AttributeError) as e:
                # Log warning instead of raising for individual bad notes
                logging.warning(f"Invalid MIDI note at index {i} ('{note}') skipped: {e}")
        return valid_notes

    def _calculate_correlation(self, hist1, hist2):
        """Calculate Pearson correlation (0-1 range). Handles NaNs/Infs/const."""
        with np.errstate(divide='ignore', invalid='ignore'):
            correlation = np.corrcoef(hist1, hist2)[0, 1]
            
        if not np.isfinite(correlation):
            return 0.0
        else:
            return (correlation + 1) / 2 # Normalize -1..1 to 0..1

    def _calculate_confidence(self, histogram, max_correlation, correlations):
        """Calculate confidence score based on correlation and heuristics."""
        confidence = float(max_correlation)
        unique_pitch_classes = np.count_nonzero(histogram)
        
        # --- Debug Logging ---
        debug_log = [f"Initial conf (max_corr): {confidence:.4f}"]
        debug_log.append(f"Unique PCs: {unique_pitch_classes}")
        # --------------------
        
        # Penalties
        if unique_pitch_classes >= 11: # Penalize heavily if input is almost fully chromatic
            penalty = 0.2
            confidence *= penalty
            debug_log.append(f"Applied Chromatic Penalty: {penalty:.2f}")
        
        if len(correlations) > 1:
            gap = max_correlation - correlations[1][2]
            debug_log.append(f"Correlation Gap: {gap:.4f}")
            # Penalize if top two candidates are close (ambiguity)
            if gap < 0.05: 
                penalty = 0.75
                confidence *= penalty
                debug_log.append(f"Applied High Ambiguity Penalty: {penalty:.2f}")
            elif gap < 0.1: 
                penalty = 0.85
                confidence *= penalty
                debug_log.append(f"Applied Moderate Ambiguity Penalty: {penalty:.2f}")
        
        final_confidence = np.clip(confidence, 0.0, 1.0)
        debug_log.append(f"Final Confidence: {final_confidence:.4f}")
        
        # Log if confidence seems low or penalties were applied
        if final_confidence < 0.7 or len(debug_log) > 3:
             logging.debug(f"Confidence Calc: {' - '.join(debug_log)}")
                
        return final_confidence

    def _update_profiling(self, analysis_time, confidence):
        """Update internal performance and confidence metrics."""
        self.total_analyses += 1
        self.total_analysis_time += analysis_time
        self.confidence_history.append(confidence)
        if analysis_time > 0.01: logging.warning(f"Slow key analysis: {analysis_time*1000:.2f}ms")

    def get_profiling_stats(self):
        """Return dict of analysis performance statistics."""
        if not self.total_analyses: return None
        avg_time = self.total_analysis_time / self.total_analyses
        avg_conf = np.mean(list(self.confidence_history)) # deque -> list
        return {'avg_analysis_time_ms': avg_time * 1000, 'avg_confidence': avg_conf}

# --- Audio Input ---
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

        # --- Add Spectrum Analyzer --- 
        # Variable to store the latest spectrum data
        self._latest_spectrum_data = [] 
        # Callback function to receive spectrum data
        def _spectrum_callback(data):
            # Spectrum gives list of lists (one per channel), we likely want the first
            if data and isinstance(data, list) and len(data) > 0:
                # --- MODIFICATION START ---
                # Expect data[0] to be a list of (freq, mag) tuples
                # Extract only the magnitude (second element) from each tuple
                try:
                    # List comprehension to get magnitudes, ensuring elements are tuples and have 2 items
                    magnitudes = [item[1] for item in data[0] if isinstance(item, tuple) and len(item) == 2]
                    self._latest_spectrum_data = magnitudes 
                    
                    # --- Updated DEBUG LOG using the processed magnitudes ---
                    if self._latest_spectrum_data:
                        latest_np = np.asarray(self._latest_spectrum_data) # Now this should be a simple float array
                        logging.debug(f"_spectrum_callback: Processed Magnitudes (len={latest_np.size}, type={latest_np.dtype}), Stored (min={np.min(latest_np):.4f}, max={np.max(latest_np):.4f})")
                    else:
                        logging.debug("_spectrum_callback: Received data[0], but it contained no valid (freq, mag) tuples.")
                except (TypeError, IndexError) as e:
                     logging.error(f"_spectrum_callback: Error processing spectrum data tuples: {e}. Raw data[0] type: {type(data[0])}, First elem type: {type(data[0][0]) if data[0] else 'N/A'}", exc_info=True)
                     self._latest_spectrum_data = [] # Clear on error
                # --- MODIFICATION END ---
            else:
                # --- NEW DEBUG LOG ---
                logging.debug(f"_spectrum_callback: Received invalid/empty data structure. Type={type(data)}, Data={data}")
                # --- END NEW DEBUG LOG ---
                self._latest_spectrum_data = [] # Clear if data structure is invalid

        # Ensure FFT size matches buffer size capabilities
        # Note: pyo's Spectrum might internally adjust size based on buffer.
        # Use overlaps=4 for standard STFT processing.
        self.spectrum = Spectrum(self.input, size=FFT_SIZE, function=_spectrum_callback)
        # ---------------------------

        logging.info(f"AudioInputProcessor initialized (buf={buffer_size}, thresh={threshold:.3f}, FFT={FFT_SIZE}).")

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

    def get_spectrum(self):
        """Returns the latest magnitude spectrum data received via callback."""
        # Return the data stored by the callback
        return self._latest_spectrum_data

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
        """Play MIDI notes as frequencies on the specified polyphonic voice type."""
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
            voice_index = i % num_voices # Cycle through voices
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
        if not num_voices: return

        # Allow custom voice selection logic (e.g., based on pattern position)
        # Default is simple modulo cycling
        voice_index = voice_selector(num_voices, getattr(self, '_internal_pos', 0))

        chosen_voice = target_voices[voice_index]
        osc = chosen_voice['osc'] # Get the oscillator object

        # --- Correctly set frequency based on oscillator type ---
        if isinstance(osc, FM):
            osc.set('carrier', freq) # Use .set('carrier', ...) for FM
            # Optional debug logging
            # logging.debug(f"Triggering FM voice {voice_index} carrier={freq:.2f}Hz")
        elif hasattr(osc, 'setFreq'): # Check if setFreq method exists for safety
            osc.setFreq(freq) # Use .setFreq(...) for other oscillators
            # Optional debug logging
            # logging.debug(f"Triggering {type(osc).__name__} voice {voice_index} freq={freq:.2f}Hz")
        else:
            logging.warning(f"Oscillator type {type(osc).__name__} has no setFreq or specific handler.")
        # -----------------------------------------------------

        chosen_voice['env'].play()
        # Minimal internal tracking for default selector (can be overridden)
        self._internal_pos = getattr(self, '_internal_pos', 0) + 1

# --- Analysis Engine ---
class AnalysisEngine:
    """
    Processes input features (onsets, pitch) to understand musical context 
    (primarily stable single notes based on Yin).
    """
    # --- REMOVED: pyo.Beat parameters, HARMONY_ANALYSIS_INTERVAL_S ---
    # --- REMOVED: Spectral/Chroma related constants/thresholds ---
    MIN_YIN_FREQ_FOR_MONO = 30 # Hz - Below typical instrument range
    # --- REMOVED: Yin stability tracking (handling directly in process_onset) ---
    # --- REMOVED: SFM constants ---

    def __init__(self, input_processor, fs):
        """Requires the InputProcessor instance."""
        self.input_processor = input_processor
        # --- REMOVED: fs dependency (not needed without spectral) ---
        self.harmonic_context = None
        # --- REMOVED: newly_confirmed_context flag ---
        # --- REMOVED: Yin stability tracking variables ---
        # --- REMOVED: last_chroma, last_sfm storage ---

        # --- REMOVED: _analysis_pattern ---
        logging.info("AnalysisEngine initialized (Simplified: Onset-driven Yin analysis).")

    def process_onset(self, timestamp):
        """
        Process a detected onset: Check current Yin pitch and amplitude 
        to update the harmonic context.
        """
        current_time = time.time() # Use the actual time of processing
        current_pitch_hz = self.input_processor.get_pitch()
        current_amp = self.input_processor.get_amplitude()
        # --- NEW DEBUG LOG --- 
        logging.debug(f"AnalysisEngine.process_onset - Raw Input: pitch={current_pitch_hz:.2f} Hz, amp={current_amp:.4f}")
        # -------------------
        # Get the threshold used by the onset detector for comparison
        # Add a small safety margin, maybe? Or use it directly. Let's use it directly for now.
        amp_threshold = self.input_processor.onset_detector.threshold

        logging.debug(f"AnalysisEngine.process_onset: time={timestamp:.4f}, pitch={current_pitch_hz:.2f}Hz, amp={current_amp:.4f}, thresh={amp_threshold:.4f}")

        new_context = None # Default to no context

        # --- Simplified Pitch/Amplitude Check ---
        # Check if amplitude is above threshold (it should be, as Thresh triggered)
        # AND if pitch is in a reasonable range.
        if current_amp >= amp_threshold and current_pitch_hz > self.MIN_YIN_FREQ_FOR_MONO:
            detected_midi = hz_to_midi(current_pitch_hz)
            if detected_midi is not None:
                # Valid pitch detected at onset
                new_context = detected_midi
                logging.debug(f"AnalysisEngine: Valid pitch {new_context} (MIDI) detected at onset.")
            else:
                logging.debug(f"AnalysisEngine: Pitch {current_pitch_hz:.2f}Hz detected, but MIDI conversion failed.")
        else:
             logging.debug(f"AnalysisEngine: Onset ignored (amp={current_amp:.4f} vs thresh={amp_threshold:.4f} or pitch={current_pitch_hz:.2f}Hz <= {self.MIN_YIN_FREQ_FOR_MONO}Hz)")
        # ----------------------------------------

        # --- Update Harmonic Context State --- 
        if self.harmonic_context != new_context:
             if new_context is not None:
                 logging.info(f"Harmonic Context Update: {new_context} (Prev: {self.harmonic_context})")
             else:
                 # Log only if context was previously set (i.e., changed to None)
                 if self.harmonic_context is not None:
                      logging.info(f"Harmonic Context Cleared (Prev: {self.harmonic_context})")
             self.harmonic_context = new_context
        # -----------------------------------

    # --- REMOVED: _run_analysis method ---

    # --- REMOVED: _analyze_harmony method ---

    def get_tempo_bpm(self):
        """Returns the estimated BPM if reliable, otherwise None."""
        return None # No beat tracking implemented

    def get_beat_phase(self):
        """Returns the current estimated beat phase if tempo is reliable, otherwise None."""
        return None # No beat tracking implemented

    def get_harmonic_context(self):
        """Returns the current harmonic context (MIDI note) or None."""
        return self.harmonic_context

    # --- REMOVED: get_last_chroma method ---
        
    # --- REMOVED: get_last_sfm method ---

    def stop(self):
        """Stop internal patterns (if any)."""
        # --- REMOVED: Stopping _analysis_pattern ---
        logging.info("AnalysisEngine stopped.")

# --- NEW: Generative Engine ---
class GenerativeEngine:
    """Decides what musical events to generate based on context and rules."""
    TARGET_OCTAVE = 4 # MIDI notes around 60
    
    def __init__(self):
        self.chord_notes_cache = {} # Simple cache for chord MIDI notes
        logging.info("GenerativeEngine initialized.")

    def _get_chord_midi_notes(self, chord_name):
        """Parses chord name (e.g., 'G#_min') and returns MIDI notes for a default octave."""
        if not chord_name or '_' not in chord_name:
            return None
            
        if chord_name in self.chord_notes_cache:
            return self.chord_notes_cache[chord_name]

        try:
            root_str, quality = chord_name.split('_')
            root_pc = PITCH_CLASSES.index(root_str)
            # Calculate root MIDI note in the target octave
            root_midi = self.TARGET_OCTAVE * 12 + root_pc

            if quality == 'maj':
                intervals = [0, 4, 7]
            elif quality == 'min':
                intervals = [0, 3, 7]
            else:
                logging.warning(f"GenerativeEngine: Unknown chord quality '{quality}'. Defaulting to major.")
                intervals = [0, 4, 7] # Default to major if quality is unknown
                
            chord_midi_notes = [root_midi + interval for interval in intervals]
            self.chord_notes_cache[chord_name] = chord_midi_notes
            return chord_midi_notes
        except (ValueError, IndexError) as e:
             logging.error(f"GenerativeEngine: Error parsing chord name '{chord_name}': {e}")
             return None

    def generate_response(self, beat_phase, harmonic_context, bpm):
        """Generate musical events based on context (MIDI note or chord).
        MODIFIED: Removed dependency on beat_phase. Triggers immediately on context.
        """
        events = []
        # --- Requires harmonic context --- REMOVED beat_phase check ---
        if harmonic_context is None:
            # logging.debug("GenEngine: Missing harmonic context, no events.") # Optional debug
            return events # Need context to generate
        # BPM is optional for current rules, but might be needed later.
        # PHASE_TOLERANCE = 0.1 # No longer needed
        # num_beats = 4 # No longer needed
        # -------------------------------------------------------------

        # --- Handle Single Note Context ---
        if isinstance(harmonic_context, (int, float)): # Check if it's a MIDI note number
            midi_note = harmonic_context
            note_freq = midi_to_hz(midi_note)
            if note_freq is None:
                 return events

            # Simple Rule: Play the note back on pluck immediately
            events.append({'synth': 'pluck', 'freq': note_freq, 'action': 'trigger'})
            logging.debug(f"GenEngine (Mono): Triggering pluck {note_freq:.2f}Hz (Context: {midi_note})")

        # --- Handle Chord Context ---
        elif isinstance(harmonic_context, str): # Check if it's a chord name string
            chord_name = harmonic_context
            chord_midi_notes = self._get_chord_midi_notes(chord_name)
            if chord_midi_notes is None or len(chord_midi_notes) == 0: # Check length > 0
                return events

            root_note = chord_midi_notes[0]
            root_freq = midi_to_hz(root_note)

            # Simple Rule: Play the root note on pluck immediately
            if root_freq:
                events.append({'synth': 'pluck', 'freq': root_freq, 'action': 'trigger'})
                logging.debug(f"GenEngine (Chord): Triggering pluck root {root_freq:.2f}Hz (Context: {chord_name})")

            # --- REMOVED OLD BEAT-PHASE BASED LOGIC ---
            # Previous Chord Logic (Bass on 1, Pluck on 2, 3, 4)
            # if (beat_phase < PHASE_TOLERANCE or beat_phase > num_beats - PHASE_TOLERANCE):
            #     if root_freq:
            #         events.append({'synth': 'bass', 'freq': root_freq, 'action': 'trigger'})
            #         logging.debug(f"GenEngine (Chord): Triggering bass root {root_freq:.2f}Hz on beat 1")
            # elif abs(beat_phase - 1.0) < PHASE_TOLERANCE:
            #      if third_freq:
            #          events.append({'synth': 'pluck', 'freq': third_freq, 'action': 'trigger'})
            #          logging.debug(f"GenEngine (Chord): Triggering pluck third {third_freq:.2f}Hz on beat 2")
            # elif abs(beat_phase - 2.0) < PHASE_TOLERANCE:
            #      if fifth_freq:
            #          events.append({'synth': 'pluck', 'freq': fifth_freq, 'action': 'trigger'})
            #          logging.debug(f"GenEngine (Chord): Triggering pluck fifth {fifth_freq:.2f}Hz on beat 3")
            # elif abs(beat_phase - 3.0) < PHASE_TOLERANCE:
            #      if root_freq:
            #          events.append({'synth': 'pluck', 'freq': root_freq, 'action': 'trigger'})
            #          logging.debug(f"GenEngine (Chord): Triggering pluck root {root_freq:.2f}Hz on beat 4")
        # -----------------------------

        return events

# --- NEW: Master Scheduler ---
class MasterScheduler:
    """
    Checks the harmonic context periodically and triggers sustained sounds 
    based on stable analysis results.
    """
    CHECK_INTERVAL_S = 0.075 # How often to check harmonic context (tune this)
    ONSET_SILENCE_THRESHOLD_S = 5.0 # Time without onsets to trigger reset attempt

    def __init__(self, figaro_instance, analysis_engine, sound_engine, generative_engine):
        # Store the main Figaro instance to access shared state/objects
        self.figaro = figaro_instance 
        self.analysis_engine = analysis_engine
        self.sound_engine = sound_engine
        self.generative_engine = generative_engine 
        self._last_played_context = None # Track the last context we triggered sound for
        self._last_context_change_time = time.time() # Timestamp of last context change

        # Pattern to periodically check context and trigger sounds
        self.check_pattern = Pattern(self._check_context_and_trigger, time=self.CHECK_INTERVAL_S)
        self.check_pattern.play()

        logging.info(f"MasterScheduler initialized (Context Check Interval: {self.CHECK_INTERVAL_S}s).")

    def _check_context_and_trigger(self):
        """Periodically checks harmonic context and triggers sounds if it changes."""
        # --- Log current smoothed amplitude --- 
        current_smoothed_amp_val = None
        current_smoothed_amp_type = None
        try:
            current_smoothed_amp = self.figaro.input_processor.smoothed_amp
            # Check if the object itself is valid before calling get()
            if current_smoothed_amp is not None:
                current_smoothed_amp_val = current_smoothed_amp.get()
                current_smoothed_amp_type = type(current_smoothed_amp_val)
            else:
                logging.warning("Scheduler tick: smoothed_amp object is None")
                
            logging.debug(f"Scheduler tick: Smoothed Amp Raw Value = {current_smoothed_amp_val}, Type = {current_smoothed_amp_type}")
            # Only log formatted value if it seems valid (e.g., a number)
            if isinstance(current_smoothed_amp_val, (int, float)):
                 logging.debug(f"Scheduler tick: Smoothed Amp Formatted = {current_smoothed_amp_val:.5f}")
            
        except Exception as e:
             logging.error(f"Scheduler tick: Error getting/logging smoothed_amp: {e}", exc_info=True)
        # --------------------------------------
        
        current_context = self.analysis_engine.get_harmonic_context()
        notes_to_play = None

        # --- Context Change Detection ---
        if current_context != self._last_played_context:
            logging.debug(f"Scheduler: Context changed from '{self._last_played_context}' to '{current_context}'")
            self._last_context_change_time = time.time() # Update timestamp on change

            if current_context is not None:
                # --- Determine MIDI notes based on context ---
                if isinstance(current_context, (int, float)): # Stable Single Note
                    midi_note = int(round(current_context))
                    if midi_note == 69: # Special case for A4
                        notes_to_play = [69, 76] # A4 + E5
                        logging.info(f"Scheduler: Detected A4 ({midi_note}), playing A+E fifth {notes_to_play}")
                    else:
                        notes_to_play = [midi_note]
                        logging.info(f"Scheduler: Detected stable note {midi_note}, playing {notes_to_play}")
                        
                elif isinstance(current_context, str): # Stable Chord
                    notes_to_play = self.generative_engine._get_chord_midi_notes(current_context)
                    if notes_to_play:
                         logging.info(f"Scheduler: Detected stable chord '{current_context}', playing {notes_to_play}")
                    else:
                         logging.warning(f"Scheduler: Could not get MIDI notes for chord '{current_context}'")
                else:
                    logging.warning(f"Scheduler: Unknown context type: {type(current_context)}")

                # --- Trigger Sound ---
                if notes_to_play:
                    # Use 'pad' voice for sustained harmony
                    self.sound_engine.play_harmony('pad', notes_to_play) 
            # else: context changed to None (silence/uncertainty), sound will decay naturally

            # --- Update last played context ---
            self._last_played_context = current_context
            
        # --- Stuck Context Timeout Check ---
        elif current_context is not None and time.time() - self._last_context_change_time > CONTEXT_STUCK_TIMEOUT_S:
             logging.warning(f"Scheduler: Context '{current_context}' seems stuck for >{CONTEXT_STUCK_TIMEOUT_S}s. Resetting.")
             # Forcibly clear the context in AnalysisEngine and the scheduler's state
             self.analysis_engine.harmonic_context = None # Directly modify if AnalysisEngine allows or add a method
             self._last_played_context = None
             self._last_context_change_time = time.time() # Reset timer
         # -----------------------------------
             
        # --- Experimental Onset Detector Reset --- 
        current_time = time.time()
        time_since_last_onset = current_time - self.figaro.last_raw_onset_time
        if time_since_last_onset > self.ONSET_SILENCE_THRESHOLD_S:
            logging.warning(f"Scheduler: No onsets detected for {time_since_last_onset:.2f}s. Attempting threshold reset.")
            try:
                original_threshold = self.figaro.input_processor.onset_detector.threshold
                # Briefly lower threshold (e.g., to half, but not below a minimum like 0.005)
                # --- Make dip slightly more aggressive --- 
                temp_threshold = max(original_threshold * 0.3, 0.003) # Lower multiplier and minimum
                # ------------------------------------------
                self.figaro.input_processor.set_threshold(temp_threshold) 
                # Maybe a tiny delay is needed? Pyo timing can be tricky.
                # time.sleep(0.01) # Let's try without sleep first
                self.figaro.input_processor.set_threshold(original_threshold) # Restore original
                # Reset the timer so we don't keep trying immediately
                self.figaro.last_raw_onset_time = current_time 
                logging.info("Scheduler: Onset detector threshold reset attempted.")
            except Exception as e:
                logging.error(f"Scheduler: Error during threshold reset: {e}")
        # -----------------------------------------
        
        # else: context hasn't changed and timeout not reached, let the current sound sustain/decay

    def stop(self):
        """Stop the scheduler's checking Pattern."""
        if self.check_pattern.isPlaying():
            self.check_pattern.stop()
        logging.info("MasterScheduler stopped.")

# --- Main Application --- 
class Figaro:
    """Main class integrating audio input, analysis, harmony, rhythm, and synthesis."""

    def __init__(self, input_device=None, output_device=None, buffer_size=256, calibration_duration=CALIBRATION_DURATION_S):
        self.server = Server(audio='portaudio', buffersize=buffer_size)
        # Explicitly set devices if provided
        if input_device is not None:
            try: self.server.setInputDevice(input_device)
            except Exception as e: logging.error(f"Failed to set input device {input_device}: {e}")
        if output_device is not None:
             try: self.server.setOutputDevice(output_device)
             except Exception as e: logging.error(f"Failed to set output device {output_device}: {e}")
             
        self.server.setMidiInputDevice(99) # Avoid MIDI conflicts
        self.server.boot()
        
        # --- Get server sampling rate --- 
        self.fs = self.server.getSamplingRate()
        logging.info(f"Server started with sampling rate: {self.fs} Hz")
        # --------------------------------
        
        self.calibrated_threshold = DEFAULT_THRESHOLD
        self.calibration_duration = calibration_duration
        self.input_processor = AudioInputProcessor(onset_callback=self.on_onset_detected, 
                                                   threshold=self.calibrated_threshold,
                                                   buffer_size=buffer_size)
        self.sound_engine = SoundEngine()
        self.analysis_engine = AnalysisEngine(input_processor=self.input_processor, fs=self.fs)
        self.generative_engine = GenerativeEngine()
        # --- Instantiate MasterScheduler (no changes needed here) --- 
        self.master_scheduler = MasterScheduler(figaro_instance=self, analysis_engine=self.analysis_engine, 
                                              sound_engine=self.sound_engine,
                                              generative_engine=self.generative_engine)
        # ------------------------------------------------
        self.last_true_onset_time = 0
        self.last_raw_onset_time = 0 # Add tracker for *any* onset callback
        logging.info("Figaro initialized.") # Simplified log

    def calibrate(self):
        """Calibrate input threshold based on ambient noise."""
        print(f"Calibrating threshold ({self.calibration_duration}s)... Be quiet.")
        # Use the Follower directly for calibration, as it's what Thresh uses
        cal_table = NewTable(length=self.calibration_duration)
        # Ensure recorder uses the exact same source as the onset detector
        recorder = TableRec(self.input_processor.smoothed_amp, table=cal_table).play() 
        time.sleep(self.calibration_duration)
        # --- Ensure recorder is stopped before accessing table --- 
        recorder.stop() 
        
        samples = np.array(cal_table.getTable())
        if samples.size > 0:
            # Calculate peak or high percentile instead of RMS for thresholding impulses
            peak_amp = np.percentile(samples, 98) # e.g., 98th percentile
            noise_floor = np.median(samples) # Estimate general noise level
            # Ensure threshold isn't ridiculously low if peak is very close to noise floor
            # Adjusted multiplier calculation for clarity
            calculated_thresh = noise_floor + (peak_amp - noise_floor) * 0.5 
            self.calibrated_threshold = max(calculated_thresh, 0.015) # Ensure a minimum reasonable threshold
            
            self.input_processor.set_threshold(self.calibrated_threshold)
            print(f"Calibration done. Noise Floor: {noise_floor:.5f}, Peak (98%): {peak_amp:.5f} => Threshold: {self.calibrated_threshold:.5f}")
        else:
            print("Calibration failed (no data). Using default threshold.")
            self.calibrated_threshold = DEFAULT_THRESHOLD # Explicitly set default if fail
            self.input_processor.set_threshold(self.calibrated_threshold)
        
        # --- Explicitly cleanup pyo objects --- 
        cal_table.reset() # Reset table data
        del recorder      # Remove reference to TableRec
        del cal_table     # Remove reference to NewTable
        # --------------------------------------

    def on_onset_detected(self, trigger_time):
        """Callback triggered by AudioInputProcessor. Feeds AnalysisEngine."""
        # --- NEW DEBUG LOG --- 
        logging.debug(f"Raw onset callback triggered at {trigger_time:.4f}")
        self.last_raw_onset_time = trigger_time # Store time of raw onset
        # -------------------
        
        # Debounce
        if trigger_time - self.last_true_onset_time < ONSET_DEBOUNCE_S:
             # logging.debug(f"Debounced onset at {trigger_time:.4f}") # Keep debug if needed
             return

        self.last_true_onset_time = trigger_time
        # --- Pass the trigger time to the analysis engine's processing method ---
        logging.debug(f"True onset detected at {trigger_time:.4f} -> AnalysisEngine.process_onset")
        self.analysis_engine.process_onset(trigger_time)
        # ----------------------------------------------------------------------

    def start(self):
        """Start the audio server, calibrate, and run main loop."""
        self.server.start()
        self.calibrate()
        print("Figaro started. Press Ctrl+C to stop.")

        try:
            loop_count = 0
            while self.server.getIsStarted():
                time.sleep(0.1) # Reduce sleep time slightly for more frequent checks
                loop_count += 1
                
                # --- Status Report every ~0.5 seconds ---
                if loop_count % 5 == 0:
                    context = self.analysis_engine.get_harmonic_context()
                    last_played = self.master_scheduler._last_played_context
                    status_parts = []
                    
                    # Detected Context
                    if context:
                        if isinstance(context, str): # Should not happen now, but keep for safety
                            context_str = f"Chord: {context}"
                        elif isinstance(context, (int, float)):
                            context_str = f"Note: {int(round(context))}" 
                        else:
                            context_str = f"Context: {context}"
                    else:
                        context_str = "Harmony: N/A"
                    status_parts.append(f"Detected: {context_str}")

                    # Playing Status
                    notes_played = self._get_notes_for_context(last_played)
                    if notes_played:
                        played_str = f"Playing: {notes_played}"
                    else:
                        played_str = "Playing: -" 
                    status_parts.append(f"{played_str}")
                    
                    # --- Use standard print with newline --- 
                    print(f"Status - { ' | '.join(status_parts) }") 
                    # -----------------------------------------

        except KeyboardInterrupt:
            print("\nStopping Figaro...")
        finally:
            self.stop()

    # Helper for status printing (add to MasterScheduler or Figaro)
    # Let's add it to Figaro class since it uses GenerativeEngine constants
    def _get_notes_for_context(self, context):
        """Helper for status display: returns the notes played for a given context."""
        if context is None: return []
        if isinstance(context, (int, float)):
            midi_note = int(round(context))
            if midi_note == 69: return [69, 76]
            else: return [midi_note]
        elif isinstance(context, str):
             # Use generative engine's helper, careful about caching/state
             # It's safer to recalculate here for display purposes
             try:
                 root_str, quality = context.split('_')
                 root_pc = PITCH_CLASSES.index(root_str)
                 root_midi = GenerativeEngine.TARGET_OCTAVE * 12 + root_pc # Access class variable
                 if quality == 'maj': intervals = [0, 4, 7]
                 elif quality == 'min': intervals = [0, 3, 7]
                 else: intervals = [0, 4, 7] # Default
                 return [root_midi + i for i in intervals]
             except:
                 return [] # Error parsing
        return []

    def stop(self):
        """Stop the audio server and analysis/scheduling engines gracefully."""
        if hasattr(self, 'master_scheduler') and self.master_scheduler: 
            self.master_scheduler.stop()
        if hasattr(self, 'analysis_engine') and self.analysis_engine:
             self.analysis_engine.stop()
        # Stop input processor? Not strictly necessary if server stops.
        if self.server.getIsBooted():
            self.server.stop()
            # Optional: Add a small delay to ensure server fully stops before script exits
            # time.sleep(0.1) 
            logging.info("Audio server stopped.")

# --- Script Execution --- 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Figaro: Interactive Harmonizer")
    parser.add_argument('-i', '--input-device', type=int, default=None, help="Audio input device ID.")
    parser.add_argument('-o', '--output-device', type=int, default=None, help="Audio output device ID.")
    parser.add_argument('--list-devices', action='store_true', help="List available audio devices and exit.")
    parser.add_argument('--buffer-size', type=int, default=256, help="Audio buffer size (samples).")
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Set logging level.')
    args = parser.parse_args()

    if args.list_devices:
        print("\nAvailable audio devices (Input IDs):")
        pa_list_devices()
        # Also list output device IDs
        print("\nAvailable audio devices (Output IDs):")
        found_out = False
        for i, device in enumerate(pa_list_devices()[1]):
             # Filter for devices belonging to the PortAudio Host API (index 0 usually)
             # and having output channels. Adjust host api index if needed based on `pa_list_devices()` output.
             if device['host api index'] == 0 and device['max output chans'] > 0: 
                  print(f"  ID: {i} - Name: {device['name']}")
                  found_out = True
        if not found_out:
             print("  (No PortAudio output devices found)")
            
        print("\nRun with '-i <ID>' to select an input device.")
        print("Run with '-o <ID>' to select an output device.")
        exit()
        
    # Set logging level from args
    log_level_from_args = getattr(logging, args.log_level.upper(), logging.INFO)
    # Configure logging (can be done once here)
    logging.basicConfig(level=log_level_from_args, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

    print(f"Starting Figaro... (Input: {args.input_device or 'Default'}, Output: {args.output_device or 'Default'}, Buffer: {args.buffer_size}, LogLevel: {args.log_level})")
    collaborator = Figaro(input_device=args.input_device, 
                        output_device=args.output_device, 
                        buffer_size=args.buffer_size)
    collaborator.start()
