import requests
import base64
import io
import soundfile as sf
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--api-url", type=str, default="https://vb5fotbuz7y7ua-3002.proxy.runpod.net/generate")
    parser.add_argument("--steps", nargs='+', type=int, default=[20, 50, 100, 200, 500])
    parser.add_argument("--duration", type=float, default=1.5)
    parser.add_argument("--output-dir", type=str, default="chirp_test_outputs")
    parser.add_argument("--timeout", type=int, default=120)
    return parser.parse_args()

args = parse_args()

PROMPT_FILENAME_SAFE = "".join(
    c if c.isalnum() or c in (' ', '-') else '_' 
    for c in args.prompt
).strip().replace(" ", "-")
PROMPT_SUBDIR_NAME = PROMPT_FILENAME_SAFE[:50]
OUTPUT_DIR = os.path.join(args.output_dir, PROMPT_SUBDIR_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)

audio_files = []

print(f"Generating audio for: \"{args.prompt}\"")
print(f"Output dir: {OUTPUT_DIR}")

for steps in args.steps:
    print(f"  Steps: {steps}...")
    payload = {"prompt": args.prompt, "duration": args.duration, "inference_steps": steps}
    response = requests.post(args.api_url, json=payload, timeout=args.timeout)
    response.raise_for_status()
    data = response.json()
    audio_base64 = data.get("audio_base64")
    audio_bytes = base64.b64decode(audio_base64)
    file_path = os.path.join(OUTPUT_DIR, f"output_{steps}_steps.wav")
    with io.BytesIO(audio_bytes) as bio:
        with open(file_path, 'wb') as f:
            f.write(bio.read())
    audio_files.append(file_path)
    print(f"    Saved: {file_path}")

print("Generating spectrogram plot...")
num_files = len(audio_files)
fig, axes = plt.subplots(1, num_files, figsize=(5 * num_files, 4), sharey=True)
fig.suptitle(f'Spectrogram Comparison: "{args.prompt}"')
axes = [axes] if num_files == 1 else axes
for i, file_path in enumerate(audio_files):
    steps = int(os.path.basename(file_path).split('_')[1])
    y, sr = librosa.load(file_path, sr=None)
    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', ax=axes[i])
    axes[i].set_title(f"{steps} Steps")
    axes[i].set_xlabel("Time (s)")
    axes[i].set_ylabel("Frequency (Hz)" if i == 0 else "")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plot_path = os.path.join(OUTPUT_DIR, "spectrogram_comparison.png")
plt.savefig(plot_path)
print(f"Plot saved: {plot_path}")
