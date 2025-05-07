#!/usr/bin/env python3
import datetime
import time
import os
import json
import torch
import contextlib
import wave
import numpy as np
import pandas as pd
import fire  # Make sure fire is imported
import gc
from pathlib import Path
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Define allowed media extensions - added MP3
MEDIA_EXTENSIONS = ['.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.m4v', '.mp3']

whisper_models = ["small", "medium", "small.en", "medium.en"]
source_languages = {
    "en": "English",
    "zh": "Chinese",
    "de": "German",
    "es": "Spanish",
    "ru": "Russian",
    "ko": "Korean",
    "fr": "French"
}
source_language_list = [key[0] for key in source_languages.items()]

# Convert media file into .wav


def convert_to_wav(media_file_path, offset=0):
    """Converts a media file (video or audio) to a 16kHz mono WAV file."""
    base_path, _ = os.path.splitext(media_file_path)
    out_path = base_path + ".wav"

    if os.path.exists(out_path):
        print(f"WAV file already exists: {out_path}, skipping conversion.")
        return out_path

    try:
        print(f"Starting conversion of '{os.path.basename(media_file_path)}' to WAV...")
        offset_args = f"-ss {offset}" if offset > 0 else ''
        # Ensure paths with spaces are quoted
        command = f'ffmpeg {offset_args} -i "{media_file_path}" -ar 16000 -ac 1 -c:a pcm_s16le "{out_path}"'
        print(f"Running command: {command}")
        result = os.system(command)
        if result != 0:
            raise RuntimeError(f"ffmpeg conversion failed with exit code {result}.")
        print(f"Conversion to WAV ready: {out_path}")
    except Exception as e:
        print(f"Error during conversion: {e}")
        # Optionally remove partially created file if conversion failed
        if os.path.exists(out_path):
            os.remove(out_path)
        raise RuntimeError(f"Error converting {media_file_path} to WAV.") from e

    return out_path


def chunked_speech_to_text(media_file_path, selected_source_lang='en', whisper_model='small.en', vad_filter=False):
    """
    Transcribes WAV files using chunked processing with Faster Whisper on GPU.
    Optimized for GTX 1660 SUPER with fixes for both CUDA allocator and exclamation mark issues.
    """

    # Only set safe environment variables - avoid CUDA allocator settings
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Fix for Intel MKL issues

    print(f"\n--- Starting GPU Transcription for: {os.path.basename(media_file_path)} ---")
    print(f'Using model: {whisper_model}')

    # Derive WAV and output paths
    base_path, _ = os.path.splitext(media_file_path)
    audio_file = base_path + ".wav"
    out_file = base_path + ".segments.json"

    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"Error: WAV file not found at {audio_file}.")

    # Check if existing transcription exists
    if os.path.exists(out_file):
        print(f"Segments file already exists: {out_file}, loading existing data.")
        try:
            with open(out_file, 'r', encoding='utf-8') as f:
                segments = json.load(f)
            print("Loaded existing segments.")
            return segments
        except Exception as e:
            print(f"Warning: Could not read existing segments file {out_file}: {e}. Retranscribing.")

    # Get audio file info
    try:
        with contextlib.closing(wave.open(audio_file, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            channels = f.getnchannels()
            duration = frames / float(rate)
        print(f"Audio duration: {datetime.timedelta(seconds=round(duration))} ({duration:.2f}s)")
    except Exception as e:
        print(f"Warning: Could not read WAV properties: {e}")
        duration = 0

    # Verify CUDA is available
    if not torch.cuda.is_available():
        print(" Warning: CUDA is not available. This function requires GPU support.")
        raise RuntimeError("This GPU-only function requires CUDA support.")

    print(f"CUDA available: GPU: {torch.cuda.get_device_name(0)}")

    # Force garbage collection
    gc.collect()
    torch.cuda.empty_cache()

    # Import faster_whisper
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("Error: faster_whisper library not found. Please install it: pip install faster-whisper")
        raise

    # Determine optimal parameters based on model
    if whisper_model in ["large-v1", "large-v2", "large-v3", "distil-large-v2", "distil-large-v3"]:
        # Smallest chunks and highest precision for large models
        chunk_size = 45  # Very small chunks for large models
        compute_type = "float32"  # Highest precision for better quality
        print("Using conservative settings for large model")
    elif whisper_model in ["medium", "medium.en", "distil-medium"]:
        # Small chunks for medium models
        chunk_size = 75  # Small chunks for medium models
        compute_type = "float32"  # High precision for better quality
        print("Using optimized settings for medium model")
    else:
        # Larger chunks for smaller models
        chunk_size = 100  # Larger chunks for smaller models
        compute_type = "float16"  # Standard for smaller models
        print("Using standard settings for small/base model")

    # For very long audio, use even smaller chunks
    if duration > 600:  # 10+ minutes
        chunk_size = max(chunk_size - 2, 3)  # Reduce chunk size but not below 3 seconds
        print(f"Long audio detected, reducing chunk size to {chunk_size}s")

    print(f"Using compute_type={compute_type}, chunk_size={chunk_size}s")

    # Create temporary directory for chunks
    import tempfile
    temp_dir = tempfile.mkdtemp()
    print(f"Created temporary directory for chunks: {temp_dir}")

    time_start = time.time()
    all_segments = []

    try:
        # Split audio into chunks using ffmpeg (more reliable)
        chunk_files = []

        for i, start_time in enumerate(range(0, int(duration), chunk_size)):
            # Create chunk file path
            chunk_file = os.path.join(temp_dir, f"chunk_{i:04d}.wav")

            # Use ffmpeg to extract this chunk
            end_time = min(start_time + chunk_size, duration)
            length = end_time - start_time

            # Extract audio segment
            cmd = f'ffmpeg -y -hide_banner -loglevel error -i "{audio_file}" -ss {start_time} -t {length} -acodec pcm_s16le -ar 16000 -ac 1 "{chunk_file}"'
            ret = os.system(cmd)

            if ret != 0:
                print(f"Error extracting chunk {i} from {start_time}s to {end_time}s")
                continue

            chunk_files.append((chunk_file, start_time))

        print(f"Split audio into {len(chunk_files)} chunks")

        # Track exclamation mark issues
        exclamation_segments = 0
        total_segments = 0

        # Process each chunk
        for i, (chunk_file, start_time) in enumerate(chunk_files):
            print(
                f"Processing chunk {i+1}/{len(chunk_files)} (starts at {datetime.timedelta(seconds=int(start_time))})")

            # Clean up GPU memory before each chunk
            gc.collect()
            torch.cuda.empty_cache()

            try:
                # Load a fresh model for each chunk (prevents CUDA issues)
                model = WhisperModel(
                    whisper_model,
                    device="cuda",
                    compute_type=compute_type,
                )

                prompt_file = ".prompt"
                default_prompt = "The transcript is about a conversation."

                if os.path.exists('.prompt'):
                    with open(prompt_file, "r") as f:
                        initial_prompt = f.read().strip() or default_prompt
                else:
                    initial_prompt = default_prompt

                # Use optimized parameters to avoid exclamation marks
                segments_raw, info = model.transcribe(
                    chunk_file,
                    beam_size=1,
                    best_of=1,
                    temperature=0,
                    compression_ratio_threshold=2.4,
                    log_prob_threshold=-1.0,
                    no_speech_threshold=0.6,
                    condition_on_previous_text=False,
                    initial_prompt=initial_prompt,
                    word_timestamps=False,
                    language=selected_source_lang,
                    vad_filter=vad_filter
                )

                # Process segments from this chunk
                chunk_segments = []
                for segment in segments_raw:
                    total_segments += 1
                    text = segment.text.strip()

                    # Check for exclamation mark issue
                    if text.count('!') > len(text) * 0.5:
                        exclamation_segments += 1
                        print(f"  [{datetime.timedelta(seconds=round(segment.start + start_time))}] Exclamation mark issue detected")
                        continue  # Skip this segment

                    segment_data = {
                        "start": segment.start + start_time,
                        "end": segment.end + start_time,
                        "text": text
                    }

                    print(f"  [{datetime.timedelta(seconds=round(segment_data['start']))} -> "
                          f"{datetime.timedelta(seconds=round(segment_data['end']))}] {segment_data['text']}")

                    chunk_segments.append(segment_data)
                    all_segments.append(segment_data)

                # Force cleanup
                del model

                print(f"  Found {len(chunk_segments)} valid segments in chunk {i+1}")

            except Exception as e:
                print(f"Error processing chunk {i+1}: {e}")
                # Continue with next chunk
                continue

            # Additional memory cleanup
            gc.collect()
            torch.cuda.empty_cache()

        # Sort all segments by start time
        all_segments.sort(key=lambda x: x["start"])

        print(f"Transcription done in {time.time() - time_start:.2f} seconds.")

        # Report on exclamation mark issues
        if exclamation_segments > 0:
            percentage = (exclamation_segments / total_segments) * 100 if total_segments > 0 else 0
            print(f"\n {exclamation_segments} segments ({percentage:.1f}%) had exclamation mark issues and were filtered out")
            if percentage > 50:
                print(
                    f"Recommendation: Try using base.en model instead of {whisper_model}, or try float32 compute type")

        if not all_segments:
            print("Warning: No valid segments were generated!")

            # If not already using base.en, suggest trying it
            if whisper_model != "base.en":
                print("Consider trying again with base.en model")

            return []

        # Save segments to JSON
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(all_segments, f, indent=2, ensure_ascii=False)
        print(f"Segments saved to: {out_file}")

    except Exception as e:
        print(f"Error during transcription: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Clean up temporary directory
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

        # Force cleanup
        gc.collect()
        torch.cuda.empty_cache()

    return all_segments


def speech_to_text(media_file_path, selected_source_lang='en', whisper_model='small.en', vad_filter=False):
    """
    Transcribes the entire WAV file at once using Faster Whisper on GPU.
    No chunking - processes the complete file in one go.

    Note: This may use more GPU memory for longer files.
    """

    # Only set safe environment variables
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Fix for Intel MKL issues

    print(f"\n--- Starting GPU Transcription for: {os.path.basename(media_file_path)} ---")
    print(f'Using model: {whisper_model}')

    # Derive WAV and output paths
    base_path, _ = os.path.splitext(media_file_path)
    audio_file = base_path + ".wav"
    out_file = base_path + ".segments.json"

    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"Error: WAV file not found at {audio_file}.")

    # Check if existing transcription exists
    if os.path.exists(out_file):
        print(f"Segments file already exists: {out_file}, loading existing data.")
        try:
            with open(out_file, 'r', encoding='utf-8') as f:
                segments = json.load(f)
            print("Loaded existing segments.")
            return segments
        except Exception as e:
            print(f"Warning: Could not read existing segments file {out_file}: {e}. Retranscribing.")

    # Get audio file info
    try:
        with contextlib.closing(wave.open(audio_file, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            channels = f.getnchannels()
            duration = frames / float(rate)
        print(f"Audio duration: {datetime.timedelta(seconds=round(duration))} ({duration:.2f}s)")

        # Warn if file is very large
        if duration > 600:  # 10 minutes
            print(f" Warning: Audio file is {duration:.1f} seconds long.")
            print(" Processing very long files without chunking may cause GPU memory issues.")
    except Exception as e:
        print(f"Warning: Could not read WAV properties: {e}")
        duration = 0

    # Verify CUDA is available
    if not torch.cuda.is_available():
        print(" Warning: CUDA is not available. This function requires GPU support.")
        raise RuntimeError("This GPU-only function requires CUDA support.")

    print(f"CUDA available: GPU: {torch.cuda.get_device_name(0)}")

    # Force garbage collection
    gc.collect()
    torch.cuda.empty_cache()

    # Import faster_whisper
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("Error: faster_whisper library not found. Please install it: pip install faster-whisper")
        raise

    # Determine optimal compute type based on model
    if whisper_model in ["medium", "medium.en", "large-v1", "large-v2", "large-v3",
                         "distil-medium", "distil-large-v2", "distil-large-v3"]:
        # For larger models, float32 gives better quality
        compute_type = "float32"
    else:
        # For smaller models, float16 is a good balance of speed and quality
        compute_type = "float16"

    print(f"Using compute_type={compute_type}")

    time_start = time.time()

    try:
        # Load model with optimal settings for your GPU
        print(f"Loading model '{whisper_model}' with compute_type={compute_type}...")
        model = WhisperModel(
            whisper_model,
            device="cuda",
            compute_type=compute_type,
            download_root=None,
            local_files_only=False,
        )

        print(f"Starting transcription of entire file ({duration:.1f} seconds)...")

        # These specific parameters help fix the exclamation mark issue
        segments_raw, info = model.transcribe(
            audio_file,
            beam_size=1,
            best_of=1,
            temperature=0,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            condition_on_previous_text=False,
            initial_prompt="The transcript is about a dialogue.",
            word_timestamps=False,
            vad_filter=vad_filter,
            language=selected_source_lang
        )

        # Process the segments
        all_segments = []
        exclamation_segments = 0
        total_segments = 0

        print("Transcription Progress:")
        for segment in segments_raw:
            total_segments += 1
            text = segment.text.strip()

            # Check for exclamation mark issue
            if text.count('!') > len(text) * 0.5:
                exclamation_segments += 1
                print(f"  [{datetime.timedelta(seconds=round(segment.start))} -> "
                      f"{datetime.timedelta(seconds=round(segment.end))}] Exclamation mark issue detected")
                continue  # Skip this segment

            segment_data = {
                "start": segment.start,
                "end": segment.end,
                "text": text
            }

            print(f"  [{datetime.timedelta(seconds=round(segment_data['start']))} -> "
                  f"{datetime.timedelta(seconds=round(segment_data['end']))}] {segment_data['text']}")

            all_segments.append(segment_data)

        print(f"Transcription done in {time.time() - time_start:.2f} seconds.")

        # Report on exclamation mark issues
        if exclamation_segments > 0:
            percentage = (exclamation_segments / total_segments) * 100
            print(f"\n {exclamation_segments} segments ({percentage:.1f}%) had exclamation mark issues and were filtered out")
            if percentage > 50:
                print(f"Recommendation: Try using base.en model instead of {whisper_model}")

        if not all_segments:
            print("Warning: No valid segments were generated! All had exclamation mark issues.")

            # If not already using base.en, try again with it
            if whisper_model != "base.en":
                print("Trying again with base.en model...")
                return speech_to_text(media_file_path, selected_source_lang, "base.en", vad_filter)

            return []

        # Save segments to JSON
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(all_segments, f, indent=2, ensure_ascii=False)
        print(f"Segments saved to: {out_file}")

    except Exception as e:
        print(f"Error during transcription: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

        # If not already using base.en, try again with it
        if whisper_model != "base.en":
            print("Trying again with base.en model...")
            return speech_to_text(media_file_path, selected_source_lang, "base.en", vad_filter)

        raise
    finally:
        # Force cleanup
        if 'model' in locals():
            del model
        gc.collect()
        torch.cuda.empty_cache()

    return all_segments

# Diarize segments from .segments.json


def speaker_diarize(media_file_path, segments, embedding_model="pyannote/embedding", embedding_size=512, num_speakers=0):
    """
    Performs speaker diarization on the transcribed segments.
    1. Generating speaker embeddings for each segment.
    2. Applying agglomerative clustering on the embeddings.
    3. Assigning speaker labels.
    """
    print(f"\n--- Starting Speaker Diarization for: {os.path.basename(media_file_path)} ---")
    if not segments:
        print("Warning: No segments provided for diarization. Skipping.")
        return None, None

    # Lazy load pyannote
    try:
        from pyannote.audio import Audio
        from pyannote.core import Segment
        from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding, Model
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics import silhouette_score
        import tqdm  # For progress bar
    except ImportError as e:
        print(
            f"Error: Required library for diarization not found ({e.name}). Please install pyannote.audio, scikit-learn, and tqdm.")
        raise

    try:
        # Load embedding model
        print(f"Loading embedding model: {embedding_model}")
        # Check for CUDA availability for embedding model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        token = ''
        with open('token.txt', 'r') as f:
            token = f.readline().strip()

        embedding_model_instance = PretrainedSpeakerEmbedding(embedding_model, device=device, use_auth_token=token)

        # Derive WAV and output JSON/CSV paths
        base_path, _ = os.path.splitext(media_file_path)
        audio_file = base_path + ".wav"
        out_file_json = base_path + ".diarize.json"
        out_file_csv = base_path + ".diarize.csv"

        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Error: WAV file not found at {audio_file} for diarization.")

        # Get duration from WAV file
        duration = 0
        try:
            with contextlib.closing(wave.open(audio_file, 'r')) as f:
                frames = f.getnframes()
                rate = f.getframerate()
                duration = frames / float(rate)
            print(f"Duration of audio file: {datetime.timedelta(seconds=round(duration))} ({duration:.2f}s)")
        except wave.Error as e:
            print(f"Warning: Could not read WAV file duration: {e}. Using segment end times.")
            # Estimate duration from the last segment if wave fails
            if segments:
                duration = segments[-1]['end']
            else:
                print("Error: Cannot determine audio duration and no segments available.")
                return None, None  # Cannot proceed without duration or segments

        # --- Embedding Calculation ---
        audio_instance = Audio()
        embeddings = np.zeros(shape=(len(segments), embedding_size))
        print("Calculating embeddings for segments...")

        def segment_embedding(segment_data):
            start = segment_data["start"]
            end = min(duration, segment_data["end"])  # Ensure end doesn't exceed duration

            # Enforce a minimum segment length for reliable embedding
            min_duration = 0.5  # seconds - adjust if needed
            if end - start < min_duration:
                padding = min_duration - (end - start)
                start = max(0, start - padding / 2)  # Don't go before 0
                end = min(duration, end + padding / 2)  # Don't go beyond duration
                # print(f'Padded segment {segment_data["id"]} from {segment_data["end"]-segment_data["start"]:.2f}s to {end-start:.2f}s')

            try:
                clip = Segment(start, end)
                waveform, sample_rate = audio_instance.crop(audio_file, clip)
                # Ensure waveform is 2D [1, num_samples] as expected by the model
                if waveform.ndim == 1:
                    waveform = waveform.unsqueeze(0)
                return embedding_model_instance(waveform)  # Shape [1, embedding_size]
            except Exception as e:
                print(f"Warning: Could not process segment {segment_data.get('id', '')} [{start:.2f}-{end:.2f}]: {e}")
                return np.zeros(embedding_size)  # Return zero vector on error

        # Add progress bar with tqdm
        for i, segment in enumerate(tqdm.tqdm(segments, desc="Embedding Segments")):
            segment['id'] = i  # Add an ID for easier debugging if needed
            embedding = segment_embedding(segment)
            if embedding is not None:
                # embedding is potentially shape [1, embedding_size], take the first row
                embeddings[i] = embedding[0] if embedding.ndim == 2 else embedding

        # Handle potential NaNs from failed embeddings (though zeros are returned now)
        embeddings = np.nan_to_num(embeddings)
        print(f'Embedding calculation complete. Shape: {embeddings.shape}')

        # Filter out segments with zero embeddings (if any failed)
        valid_indices = np.where(np.abs(embeddings).sum(axis=1) > 1e-5)[0]
        if len(valid_indices) < len(segments):
            print(
                f"Warning: {len(segments) - len(valid_indices)} segments had near-zero embeddings and were excluded from clustering.")
            if len(valid_indices) == 0:
                print("Error: No valid embeddings generated. Cannot perform clustering.")
                return None, None
            segments_for_clustering = [segments[i] for i in valid_indices]
            embeddings_for_clustering = embeddings[valid_indices, :]
        else:
            segments_for_clustering = segments
            embeddings_for_clustering = embeddings

        # --- Clustering ---
        if num_speakers == 0:
            # Find the best number of speakers using silhouette score (if enough segments)
            min_clusters = 2
            # Limit max clusters based on available segments, e.g., max 10 or half the segments
            max_clusters = min(10, len(segments_for_clustering) // 2)
            score_num_speakers = {}

            if max_clusters < min_clusters:
                print("Warning: Not enough distinct segments to perform automatic speaker count detection. Assuming 1 speaker.")
                best_num_speaker = 1
            else:
                print(f"Finding optimal number of speakers between {min_clusters} and {max_clusters}...")
                for k in range(min_clusters, max_clusters + 1):
                    try:
                        clustering = AgglomerativeClustering(
                            n_clusters=k, linkage='ward').fit(embeddings_for_clustering)
                        # Silhouette score requires at least 2 labels and less than n_samples-1 labels
                        if len(
                                set(clustering.labels_)) > 1 and len(
                                set(clustering.labels_)) < len(embeddings_for_clustering):
                            score = silhouette_score(embeddings_for_clustering, clustering.labels_, metric='euclidean')
                            score_num_speakers[k] = score
                            print(f"  Num Speakers: {k}, Silhouette Score: {score:.4f}")
                        else:
                            print(f"  Num Speakers: {k}, Silhouette Score: N/A (invalid label count)")
                    except Exception as e:
                        print(f"  Error clustering for {k} speakers: {e}")

                if score_num_speakers:
                    best_num_speaker = max(score_num_speakers, key=score_num_speakers.get)
                    print(
                        f"Optimal number of speakers determined: {best_num_speaker} (Score: {score_num_speakers[best_num_speaker]:.4f})")
                else:
                    print("Warning: Could not determine optimal number of speakers automatically. Defaulting to 1.")
                    best_num_speaker = 1  # Fallback if scores couldn't be calculated
        else:
            print(f"Using pre-defined number of speakers: {num_speakers}")
            best_num_speaker = num_speakers

        # --- Assign Speaker Labels ---
        if best_num_speaker > 0:
            # If best_num_speaker is 1, no need to cluster, assign SPEAKER 1 to all
            if best_num_speaker == 1:
                labels = [0] * len(segments_for_clustering)
                print("Assigning single speaker label (SPEAKER 1) to all segments.")
            elif len(segments_for_clustering) < best_num_speaker:
                print(
                    f"Warning: Fewer segments ({len(segments_for_clustering)}) than requested speakers ({best_num_speaker}). Assigning labels based on available segments.")
                # Assign unique label to each segment up to best_num_speaker
                labels = list(range(len(segments_for_clustering)))
            else:
                print(f"Clustering embeddings into {best_num_speaker} speakers...")
                clustering = AgglomerativeClustering(n_clusters=best_num_speaker,
                                                     linkage='ward').fit(embeddings_for_clustering)
                labels = clustering.labels_
                print("Clustering complete.")
        else:
            print("Error: Invalid number of speakers determined (<=0). Cannot assign labels.")
            return None, None  # Cannot proceed

        # Assign labels back to the original segments list
        # Create a mapping from the index in the filtered list to the label
        label_map = {valid_indices[i]: labels[i] for i in range(len(valid_indices))}

        # Assign labels, defaulting for segments that were excluded
        speaker_label_prefix = "SPEAKER_"  # Changed from "SPEAKER " for consistency
        num_digits = len(str(best_num_speaker))  # For formatting like SPEAKER_01, SPEAKER_02
        for i in range(len(segments)):
            if i in label_map:
                # Format label with leading zeros if needed (e.g., SPEAKER_01 instead of SPEAKER_1)
                formatted_label = str(label_map[i] + 1).zfill(num_digits)
                segments[i]["speaker"] = f"{speaker_label_prefix}{formatted_label}"
            else:
                # Assign a default label like SPEAKER_UNKNOWN or skip
                segments[i]["speaker"] = f"{speaker_label_prefix}UNKNOWN"
                print(f"Segment {i} assigned UNKNOWN speaker label (likely due to embedding error).")

        # --- Save Diarized Segments (JSON) ---
        try:
            with open(out_file_json, 'w', encoding='utf-8') as f:
                json.dump(segments, f, indent=2, ensure_ascii=False)
            print(f"Diarized segments saved to: {out_file_json}")
        except Exception as e:
            print(f"Error saving diarized JSON file {out_file_json}: {e}")

        # --- Create and Save Combined CSV Output ---
        print("Creating combined CSV output...")

        def convert_time(secs):
            return str(datetime.timedelta(seconds=round(secs)))

        output_data = []
        current_speaker = None
        current_text = ""
        current_start = 0

        for i, segment in enumerate(segments):
            speaker = segment.get("speaker", f"{speaker_label_prefix}UNKNOWN")
            text = segment.get("text", "").strip()
            start = segment["start"]
            end = segment["end"]

            if speaker != current_speaker and current_text:
                # End the previous block
                output_data.append({
                    'Start': convert_time(current_start),
                    'End': convert_time(segments[i-1]["end"]),  # Use previous segment's end
                    'Speaker': current_speaker,
                    'Text': current_text.strip()
                })
                current_text = ""  # Reset text

            if not current_text:  # Start of a new block (or first segment)
                current_speaker = speaker
                current_start = start

            current_text += text + " "  # Add space between segments

        # Add the last block
        if current_text:
            output_data.append({
                'Start': convert_time(current_start),
                'End': convert_time(segments[-1]["end"]),  # Use last segment's end
                'Speaker': current_speaker,
                'Text': current_text.strip()
            })

        if output_data:
            df_results = pd.DataFrame(output_data)
            try:
                df_results.to_csv(out_file_csv, index=False, encoding='utf-8')
                print(f"Combined CSV output saved to: {out_file_csv}")
                return df_results, out_file_csv
            except Exception as e:
                print(f"Error saving CSV file {out_file_csv}: {e}")
                # Return the dataframe even if saving fails
                return df_results, None
        else:
            print("No data to write to CSV.")
            return None, None

    except FileNotFoundError as e:
        print(e)
        raise  # Reraise FileNotFoundError
    except Exception as e:
        import traceback
        print(f"Error Running Speaker Diarization: {e}")
        traceback.print_exc()  # Print detailed traceback
        raise RuntimeError(f"Error during speaker diarization for {media_file_path}") from e


# --- Helper functions for checking existing files ---
def has_existing_transcription(file_path):
    """Check if a file already has a transcription file."""
    base_path, _ = os.path.splitext(file_path)
    segments_file = base_path + ".segments.json"
    return os.path.exists(segments_file)

# --- Main Execution Logic ---


def main(input_folder: str, num_speakers: int = 0, whisper_model: str = "small.en", offset: int = 0, vad_filter: bool = False):
    """
    Processes media files in a folder: converts to WAV, transcribes, and diarizes.

    Args:
        input_folder (str): Path to the folder containing media files (video or audio).
        num_speakers (int): Number of speakers for diarization (0 for auto-detect).
        whisper_model (str): Name of the Faster Whisper model to use.
        offset (int): Start processing the video/audio from this offset (in seconds).
        vad_filter (bool): Enable VAD (Voice Activity Detection) filter in Whisper.
    """
    print(f"Starting processing for folder: {input_folder}")
    print(
        f"Parameters: num_speakers={num_speakers}, whisper_model={whisper_model}, offset={offset}, vad_filter={vad_filter}")

    if not os.path.isdir(input_folder):
        print(f"Error: Input folder not found or is not a directory: {input_folder}")
        return

    processed_files = 0
    skipped_files = 0
    error_files = 0

    # Iterate through all files in the input folder
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)

        # Check if it's a file and has a valid media extension
        if os.path.isfile(file_path) and any(filename.lower().endswith(ext) for ext in MEDIA_EXTENSIONS):
            print(f"\n>>> Processing media file: {filename} <<<")

            # Check if transcription already exists
            if has_existing_transcription(file_path):
                print(f"Transcription already exists for {filename}, skipping transcription and diarization.")
                skipped_files += 1
                continue

            try:
                # 1. Convert to WAV
                wav_path = convert_to_wav(file_path, offset)

                # 2. Transcribe (Speech to Text)
                segments = chunked_speech_to_text(file_path, whisper_model=whisper_model, vad_filter=vad_filter)

                # 3. Speaker Diarization
                if segments:  # Only diarize if transcription produced segments
                    _, save_path = speaker_diarize(file_path, segments, num_speakers=num_speakers)
                    if save_path:
                        print(f"--- Diarization complete for {filename}. Output CSV: {save_path} ---")
                    else:
                        print(f"--- Diarization completed for {filename}, but CSV save failed or no data. ---")
                else:
                    print(f"--- Skipping diarization for {filename} due to empty transcription. ---")

                processed_files += 1

            except FileNotFoundError as e:
                print(f"Error processing {filename}: {e}. Skipping file.")
                error_files += 1
            except RuntimeError as e:
                print(f"Runtime error processing {filename}: {e}. Skipping file.")
                error_files += 1
            except Exception as e:
                import traceback
                print(f"An unexpected error occurred while processing {filename}: {e}")
                traceback.print_exc()
                error_files += 1
            print(f">>> Finished processing: {filename} <<<")

        elif os.path.isfile(file_path) and filename.lower().endswith('.wav'):
            print(
                f"--- Found WAV file: {filename}. Skipping direct processing (processed via media conversion if applicable). ---")
        # Optionally add handling for other file types or subdirectories if needed

    print(f"\n=== Processing Summary ===")
    print(f"Total media files processed: {processed_files}")
    print(f"Total files skipped (already had transcription): {skipped_files}")
    print(f"Total files skipped due to errors: {error_files}")
    print(f"Processed folder: {input_folder}")
    print("==========================")


if __name__ == "__main__":
    # Make sure dependencies are mentioned to the user if they are missing
    try:
        import faster_whisper
        import pyannote.audio
        import sklearn
        import pandas
        import torch
        import tqdm  # Added tqdm check
    except ImportError as e:
        print(f"Error: Missing required library '{e.name}'.")
        print("Please install all necessary libraries:")
        print("pip install faster-whisper pyannote.audio scikit-learn pandas torch tqdm")
        # For CUDA support with torch: refer to official PyTorch installation instructions.
        # For ffmpeg: Ensure ffmpeg is installed and available in your system's PATH.
        exit(1)

    # Use Fire for command-line argument parsing
    fire.Fire(main)
