#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "accelerate",
#     "diffusers",
#     "huggingface_hub[cli]",
#     "fastapi",
#     "soundfile",
#     "torch",
#     "torchsde",
#     "transformers",
#     "uvicorn",
# ]
# ///
import torch
import soundfile as sf
import base64
import io
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from diffusers import StableAudioPipeline

app = FastAPI()

# Add CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load model with proper settings
pipe = StableAudioPipeline.from_pretrained(
    "stabilityai/stable-audio-open-1.0", 
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

# Add torch.compile optimizations
pipe.transformer.to(memory_format=torch.channels_last)
pipe.vae.to(memory_format=torch.channels_last)
pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune", fullgraph=True)
pipe.vae.decode = torch.compile(pipe.vae.decode, mode="max-autotune", fullgraph=True)

class AudioRequest(BaseModel):
    prompt: str
    negative_prompt: str = "Low quality."
    seed: int = 0
    duration: float = 3.0
    inference_steps: int = 50

@app.post("/generate")
async def generate_audio(request: AudioRequest):
    generator = torch.Generator("cuda").manual_seed(request.seed)
    
    audio = pipe(
        request.prompt,
        negative_prompt=request.negative_prompt,
        num_inference_steps=request.inference_steps,
        audio_end_in_s=request.duration,
        num_waveforms_per_prompt=1,
        generator=generator,
    ).audios
    
    # Improved WAV generation
    output = audio[0].T.float().cpu().numpy()
    buffer = io.BytesIO()
    sf.write(
        buffer, 
        output, 
        samplerate=pipe.vae.sampling_rate,
        format='WAV',
        subtype='PCM_16',
        endian='FILE'
    )
    buffer.seek(0)
    
    audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    
    return {
        "status": "success",
        "prompt": request.prompt,
        "duration": request.duration,
        "sample_rate": pipe.vae.sampling_rate,
        "audio_base64": audio_base64
    }

if __name__ == "__main__":
    import uvicorn
    # Set host to 0.0.0.0 to allow external connections
    uvicorn.run(app, host="0.0.0.0", port=3002)