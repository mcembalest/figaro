from pyo import *
import time
import numpy as np

class Figaro:
    def __init__(self, input_device=None, calibration_duration=3):
        self.server = Server(audio='portaudio')
        if input_device is not None:
            self.server.setInputDevice(input_device)
        self.server.setMidiInputDevice(99)
        self.server.boot()
        self.input = Input(chnl=0, mul=1)
        self.pitch_detector = Yin(self.input, minfreq=50, maxfreq=2000, cutoff=0.5)
        self.smooth_pitch = Port(self.pitch_detector, risetime=0.05, falltime=0.05)
        self.amp_follower = Follower(self.input, freq=20)
        self.calibration_duration = calibration_duration
        self.threshold = 0.05
        self.calibrated = False
        self.create_harmonizer()
        self.trigger = Thresh(self.amp_follower, self.threshold)
        self.trigger_func = TrigFunc(self.trigger, self.on_note_detected)

    def create_harmonizer(self):
        """Create the harmonizer voices"""
        self.harmonizer_oscs = []
        env1 = Adsr(attack=0.02, decay=0.1, sustain=0.7, release=0.5, dur=0.5, mul=0.2)
        osc1 = SuperSaw(freq=440, detune=0.3, mul=env1)
        self.harm1 = {"env": env1, "osc": osc1}
        self.harmonizer_oscs.append(osc1)
        env2 = Adsr(attack=0.02, decay=0.2, sustain=0.6, release=0.7, dur=0.5, mul=0.15)
        osc2 = SuperSaw(freq=440, detune=0.4, mul=env2)
        self.harm2 = {"env": env2, "osc": osc2}
        self.harmonizer_oscs.append(osc2)
        env3 = Adsr(attack=0.01, decay=0.1, sustain=0.5, release=0.6, dur=0.5, mul=0.12)
        osc3 = SuperSaw(freq=880, detune=0.5, mul=env3)
        self.harm3 = {"env": env3, "osc": osc3}
        self.harmonizer_oscs.append(osc3)
        self.harm_mix = Mix(self.harmonizer_oscs, voices=1)
        self.reverb = Freeverb(self.harm_mix, size=0.85, damp=0.4, bal=0.15)
        self.reverb.out()

    def calibrate(self):
        """Calibrate the amplitude threshold based on room noise"""
        print(f"Calibrating for {self.calibration_duration} seconds... Please be quiet.")
        calibration_table = NewTable(length=self.calibration_duration, chnls=1)
        self.room_recorder = TableRec(self.amp_follower,
                                    table=calibration_table,
                                    fadetime=0.01)
        self.room_recorder.play()
        time.sleep(self.calibration_duration)
        room_tone_data = calibration_table.getTable()
        if len(room_tone_data) > 0:
            room_rms = np.sqrt(np.mean(np.array(room_tone_data)**2))
            self.threshold = max(room_rms * 5.0, 0.05) # 5x room tone or minimum 0.05
            print(f"Calibration complete. Ambient level: {room_rms:.5f}, Threshold: {self.threshold:.5f}")
            self.trigger.setThreshold(self.threshold)
            self.calibrated = True
        else:
            print("Calibration failed - no data recorded. Using default threshold.")

    def on_note_detected(self):
        """Respond when a note is detected"""
        current_pitch = self.smooth_pitch.get()
        current_amp = self.amp_follower.get()
        if current_pitch < 50 or current_amp < self.threshold:
            return # Filter out unreliable pitch detection
        harmony = self.generate_harmony(current_pitch)
        self.harm1["osc"].setFreq(harmony[0])
        self.harm2["osc"].setFreq(harmony[1])
        self.harm3["osc"].setFreq(harmony[2])
        self.harm1["env"].play()
        self.harm2["env"].play()
        self.harm3["env"].play()

    def generate_harmony(self, root_freq):
        """Generate harmony notes based on the root frequency"""
        root_freq = float(root_freq)
        root_midi = 69 + 12 * np.log2(root_freq / 440.0)
        time_based_choice = int(time.time() * 0.2) % 3  # Changes every 5 seconds
        if time_based_choice == 0:  # Major
            harmony_midi = [root_midi, root_midi + 4, root_midi + 12]  # Root, major 3rd, octave
        elif time_based_choice == 1:  # Minor
            harmony_midi = [root_midi, root_midi + 3, root_midi + 12]  # Root, minor 3rd, octave
        else:  # Sus4
            harmony_midi = [root_midi, root_midi + 5, root_midi + 12]  # Root, perfect 4th, octave
        harmony_freqs = [float(440.0 * (2.0 ** ((note - 69) / 12.0))) for note in harmony_midi]
        return harmony_freqs

    def start(self):
        """Start the server and begin processing"""
        self.server.start()
        self.calibrate()
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.server.stop()

if __name__ == "__main__":
    print("Available audio devices:")
    pa_list_devices()
    device_input = input("\nSelect input device number (press Enter for default): ").strip()
    input_device = int(device_input) if device_input.isdigit() else None
    collaborator = Figaro(input_device=input_device)
    collaborator.start()