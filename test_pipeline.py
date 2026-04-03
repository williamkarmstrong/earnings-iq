import argparse
import sys
import os
from speech import transcribe_audio, map_speakers, tokenize_audio_text

def test_pipeline(audio_file_path):
    print(f"Starting test pipeline on: {audio_file_path}")
    
    # 1. Transcribe
    print("\n--- Transcribing Audio ---")
    transcription = transcribe_audio(audio_file_path)
    print(f"Transcription text length: {len(transcription['text'])} characters")
    print(f"Successfully found {len(transcription.get('segments', []))} text segments.")
    
    # 2. Map Speakers
    print("\n--- Mapping Speakers (Diarization) ---")
    
    # ADD HUGGING FACE API KEY WHEN TESTING
    hf_token = ""

    if not hf_token:
        print("Warning: HUGGING_FACE_API_KEY is not set. Speaker mapping may fail.")
        
    mapped_segments = map_speakers(audio_file_path, transcription)
    # Give a quick sample of the first 3 mapped segments
    print("Sample mapped segments (first 3):")
    for seg in mapped_segments[:3]:
        print(f"[{seg['start']:.2f}s -> {seg['end']:.2f}s] {seg.get('speaker', 'UNKNOWN')}: {seg['text']}")
        
    # 3. Tokenize
    print("\n--- Tokenizing Final Text ---")
    tokens = tokenize_audio_text(transcription['text'])
    print(f"Total tokens generated: {len(tokens)}")
    print(f"First 10 tokens: {tokens[:10]}")

    print("\nTest pipeline complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Pipeline for Earnings-IQ Audio processing")
    parser.add_argument("audio_path", help="Path to a small .mp3 or .wav test file")
    args = parser.parse_args()
    
    if not os.path.exists(args.audio_path):
        print(f"Error: File '{args.audio_path}' does not exist.")
    else:
        test_pipeline(args.audio_path)
