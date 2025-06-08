from pyo import *
from pynput import keyboard
import argparse

ROW1_KEYS = 'zxcvbnm,./'
ROW2_KEYS = 'asdfghjkl;\'\\'
ROW3_KEYS = 'qwertyuiop[]'
ROW4_KEYS = '1234567890-='
ALL_ROWS = [ROW1_KEYS, ROW2_KEYS, ROW3_KEYS, ROW4_KEYS]
DEFAULT_ROOT_NOTE_MIDI = 40
ATTACK, DECAY, SUSTAIN, RELEASE = 0.01, 0.1, 0.7, 0.3
NOTE_VOLUME = 0.3

active_notes = {}
key_to_midi = {}

def midi_to_freq(midi_note):
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))

def build_key_map(root_note_midi):
    global key_to_midi
    key_to_midi.clear()
    for row_index, row_keys in enumerate(ALL_ROWS):
        row_start_midi = root_note_midi + row_index * 12
        for key_index, key_char in enumerate(row_keys):
            key_to_midi[key_char] = row_start_midi + key_index

def on_press(key):
    global active_notes
    try:
        char = key.char
        if char in key_to_midi and char not in active_notes:
            midi_note = key_to_midi[char]
            freq = midi_to_freq(midi_note)
            env = Adsr(attack=ATTACK, decay=DECAY, sustain=SUSTAIN, release=RELEASE, dur=0, mul=NOTE_VOLUME)
            sine = Sine(freq=freq+mod, mul=env)
            sine.out()
            env.play()
            active_notes[char] = {'sine': sine, 'env': env}
    except AttributeError:
        pass

def on_release(key):
    global active_notes
    try:
        char = key.char
        if char in active_notes:
            active_notes[char]['env'].stop()
            del active_notes[char]
    except AttributeError:
        if key == keyboard.Key.esc:
            return False

def main():
    global s, mod
    parser = argparse.ArgumentParser(description="QWERTY Keyboard Synthesizer")
    parser.add_argument("--root", type=int, default=DEFAULT_ROOT_NOTE_MIDI, help="MIDI root note number")
    args = parser.parse_args()
    
    build_key_map(args.root)
    s = Server().boot()
    mod = Sine(freq=6, mul=7)
    s.start()
    
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    
    try:
        listener.join()
    except KeyboardInterrupt:
        if listener.is_alive():
            listener.stop()
    finally:
        if s and s.getIsStarted():
            s.stop()

if __name__ == "__main__":
    main()