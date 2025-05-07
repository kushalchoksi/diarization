#!/usr/bin/env python3
"""
Audio normalization helper script to fix low volume audio files 
that cause the '!!!' exclamation mark issue in whisper transcription.

This script:
1. Analyzes audio files to check volume levels
2. Normalizes audio that's too quiet using ffmpeg
3. Can be used as a standalone script or imported into your main script

Usage:
  python normalize_audio.py input.wav [output.wav] [--threshold 0.1] [--target-level -16]
  
If output.wav is not specified, it will create a normalized version with "_normalized" suffix.
"""

import subprocess
import wave
import numpy as np
import argparse
from pathlib import Path

def analyze_audio(wav_path):
    """Analyze audio file and return stats about its volume."""
    try:
        with wave.open(wav_path, 'rb') as wf:
            # Get basic properties
            frames = wf.getnframes()
            rate = wf.getframerate()
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            duration = frames / float(rate)
            
            # Read entire file for analysis
            wf.setpos(0)
            buffer = wf.readframes(frames)
            
            # Convert to numpy array based on bit depth
            if sample_width == 2:  # 16-bit audio
                audio_data = np.frombuffer(buffer, dtype=np.int16)
                max_possible = 32768.0
            elif sample_width == 4:  # 32-bit audio
                audio_data = np.frombuffer(buffer, dtype=np.int32)
                max_possible = 2147483648.0
            else:  # 8-bit audio
                audio_data = np.frombuffer(buffer, dtype=np.uint8)
                max_possible = 256.0
            
            # Calculate stats
            abs_data = np.abs(audio_data)
            max_amplitude = np.max(abs_data)
            mean_amplitude = np.mean(abs_data)
            rms = np.sqrt(np.mean(np.square(audio_data.astype(float))))
            
            # Normalized values (0.0 to 1.0)
            max_norm = max_amplitude / max_possible
            rms_norm = rms / max_possible
            
            # dB values
            if max_amplitude > 0:
                max_db = 20 * np.log10(max_norm)
            else:
                max_db = -96.0  # Near silence
                
            if rms > 0:
                rms_db = 20 * np.log10(rms_norm)
            else:
                rms_db = -96.0  # Near silence
            
            return {
                "duration": duration,
                "sample_rate": rate,
                "channels": channels,
                "bit_depth": sample_width * 8,
                "max_amplitude": max_amplitude,
                "max_norm": max_norm,
                "max_db": max_db,
                "rms_norm": rms_norm,
                "rms_db": rms_db,
                "mean_amplitude": mean_amplitude
            }
    except Exception as e:
        print(f"Error analyzing audio file: {e}")
        return None

def needs_normalization(stats, threshold=0.1):
    """Determine if audio needs normalization based on stats and threshold."""
    if not stats:
        return True  # If analysis failed, assume it needs normalization
    
    # Different ways to determine if normalization is needed
    if stats["max_norm"] < threshold:
        return True
    
    if stats["rms_db"] < -30:  # Very quiet audio
        return True
        
    return False

def normalize_audio(input_path, output_path=None, target_level=-16):
    """
    Normalize audio using ffmpeg to fix volume issues.
    
    Args:
        input_path: Path to input WAV file
        output_path: Path to output WAV file (if None, creates one with _normalized suffix)
        target_level: Target loudness level in dB (default: -16 LUFS, which is a good level for speech)
    
    Returns:
        Path to the normalized file if successful, None otherwise
    """
    if output_path is None:
        # Create output path with _normalized suffix
        input_path_obj = Path(input_path)
        output_path = str(input_path_obj.with_stem(f"{input_path_obj.stem}_normalized"))
    
    try:
        # Get audio stats first
        stats = analyze_audio(input_path)
        if stats:
            print(f"Original audio stats:")
            print(f"  Duration: {stats['duration']:.2f}s")
            print(f"  Max volume: {stats['max_db']:.2f} dB")
            print(f"  RMS volume: {stats['rms_db']:.2f} dB")
        
        if not needs_normalization(stats, threshold=0.1):
            print(f"Audio already has good volume levels (max: {stats['max_db']:.2f} dB). Skipping normalization.")
            return input_path
            
        print(f"Normalizing audio to target level of {target_level} dB LUFS...")
        
        # Run ffmpeg with loudnorm filter for proper audio normalization
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-af", f"loudnorm=I={target_level}:LRA=11:TP=-1.5:print_format=summary",
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", 
            output_path
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error normalizing audio: {result.stderr}")
            return None
            
        # Verify the result
        new_stats = analyze_audio(output_path)
        if new_stats:
            print(f"Normalized audio stats:")
            print(f"  Duration: {new_stats['duration']:.2f}s")
            print(f"  Max volume: {new_stats['max_db']:.2f} dB")
            print(f"  RMS volume: {new_stats['rms_db']:.2f} dB")
            
            improvement = new_stats['max_db'] - stats['max_db']
            print(f"Volume increased by approximately {improvement:.2f} dB")
            
        return output_path
        
    except Exception as e:
        print(f"Error during audio normalization: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description="Normalize audio files to fix low volume")
    parser.add_argument("input", help="Input WAV file")
    parser.add_argument("output", nargs='?', default=None, help="Output WAV file (optional)")
    parser.add_argument("--threshold", type=float, default=0.1, 
                        help="Normalization threshold (0.0-1.0). Lower values trigger normalization more often")
    parser.add_argument("--target-level", type=float, default=-16, 
                        help="Target loudness level in LUFS (typical range: -24 to -14)")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Only analyze the audio without normalizing")
    args = parser.parse_args()
    
    print(f"Analyzing audio file: {args.input}")
    stats = analyze_audio(args.input)
    
    if stats:
        print("\nAudio Analysis Results:")
        print(f"  Duration: {stats['duration']:.2f} seconds")
        print(f"  Sample rate: {stats['sample_rate']} Hz")
        print(f"  Channels: {stats['channels']}")
        print(f"  Bit depth: {stats['bit_depth']} bits")
        print(f"  Maximum amplitude: {stats['max_amplitude']} ({stats['max_db']:.2f} dB)")
        print(f"  RMS level: {stats['rms_norm']:.6f} ({stats['rms_db']:.2f} dB)")
        
        if args.analyze_only:
            if needs_normalization(stats, args.threshold):
                print("\n This audio has low volume and would benefit from normalization.")
                print(f"  Run this script without --analyze-only to normalize it.")
            else:
                print("\n✓ This audio has good volume levels and doesn't need normalization.")
            return
    else:
        print("Failed to analyze audio. Will attempt normalization anyway.")
    
    normalized_path = normalize_audio(args.input, args.output, args.target_level)
    
    if normalized_path:
        print(f"\n✓ Audio normalization complete: {normalized_path}")
        if normalized_path != args.input:
            print(f"  Original file preserved: {args.input}")
    else:
        print("\nX Audio normalization failed. Check error messages above.")

if __name__ == "__main__":
    main()