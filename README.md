# Diarization using Faster Whisper

## What it does

1) Converts video / audio files to WAV format
2) Transcribes speech using Faster Whisper (GPU-accelerated)
3) Identifies different speakers in the audio
4) Outputs formatted transcripts with speaker labels

## Requirements
- Python 3.7+
- PyTorch with CUDA Support (I personally used cuda 12.6 on Windows)
- NVIDIA GPU (Ran Whisper distill-large-v3 with ease)
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

#### Large models (large-v1, large-v2, large-v3):

Use very small chunks (45 seconds)
Higher precision (float32)
More memory-intensive but more accurate

#### Medium models (medium, medium.en):

Use medium chunks (75 seconds)
Higher precision (float32)
Good balance of accuracy and resource usage

#### Small models (small, small.en, base):

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

