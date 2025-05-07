# Diarization using Faster Whisper

### Input
---
![Audio sample](test/audio.wav)
---

### Outputs
```json
...
  {
    "start": 10.24,
    "end": 11.4,
    "text": "Do you think we can get back to that?",
    "id": 2,
    "speaker": "SPEAKER_2"
  },
  {
    "start": 12.4,
    "end": 14.8,
    "text": "Not for now. We can, we have to try.",
    "id": 3,
    "speaker": "SPEAKER_1"
  },
  {
    "start": 14.96,
    "end": 21.66,
    "text": "I mean, I will certainly try because I'm a big believer, that football is changing and you need to adapt to every situation",
    "id": 4,
    "speaker": "SPEAKER_1"
  },
...
```

| Start | End | Speaker | Transcription |
|-------|-----|---------|------|
0:00:00|0:00:10|SPEAKER_1|"You could feel, you could smell, you could sense the next step, you could smell the play. You could smell if the other team was, you know, slowing down."
0:00:10|0:00:11|SPEAKER_2|Do you think we can get back to that?
0:00:12|0:00:28|SPEAKER_1|"Not for now. We can, we have to try. I mean, I will certainly try because I'm a big believer, that football is changing and you need to adapt to every situation and you need to adapt to what's happening now. I mean, you cannot get away from this and you cannot underestimate the..."


## What it does

1) Converts video / audio files to WAV format
2) Transcribes speech using Faster Whisper (GPU-accelerated)
3) Identifies different speakers in the audio
4) Outputs formatted transcripts with speaker labels

## Requirements
- Python 3.7+
- PyTorch with CUDA Support (I personally used cuda 12.6 on Windows)
- NVIDIA GPU (Ran Whisper distill-large-v3 with ease on GeForce GTX 1660)
- ffmpeg (The 🐐)

## Setup
- Install the dependencies using `requirements.txt` 
- Create a `token.txt` file containing your HuggingFace token to pull any faster-whisper models that aren't cached on your system (will probably ask you to accept pyannote model license)
- (Optional) Create a .prompt file with custom transcription

## Usage
```shell
python diarize.py input_folder
OR
python diarize.py input_folder --num_speakers=2 --whisper_model=medium.en --vad_filter=True
```

## Model Size and Chunking
The tool adjusts processing settings based on model size:

### Large models (large-v1, large-v2, large-v3):

Use very small chunks (45 seconds)
Higher precision (float32)
More memory-intensive but more accurate

### Medium models (medium, medium.en):

Use medium chunks (75 seconds)
Higher precision (float32)
Good balance of accuracy and resource usage

### Small models (small, small.en, base):

Use larger chunks (100 seconds)
Lower precision (float16)
Less resource-intensive but potentially less accurate

For very long audio (10+ minutes), chunk sizes are automatically reduced further to prevent memory issues.

## Troubleshooting

If you encounter the "exclamation mark issue" (text filled with !!!), try:
- Normalizing the audio using the normalization script (it'll amplify the volume level)
- Usually happens when trying to use a larger model than what your GPU VRAM can handle, so lower the chunk size
- Try using a smaller model
- Get a better GPU 

