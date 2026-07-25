"""
Video Transcriber System
Extracts exact voice (transcription) from video files.
"""

import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from transcriber.logger import setup_logger
from transcriber.video_processor import take_video_input, convert_video_to_audio
from transcriber.speech_understanding import transcribe_audio

def print_header():
    print("""
=================================================
   Video Transcriber System
=================================================
""")

def process_pipeline():
    """Process video to extract transcription."""
    print("\n" + "="*50)
    print("STEP 1: Video Input")
    print("="*50)
    
    video_path = take_video_input()
    
    if video_path is None:
        print("Video import failed.")
        return False
    
    print("\n" + "="*50)
    print("STEP 2: Audio Extraction")
    print("="*50)
    audio_path = convert_video_to_audio(video_path, force=True)
    
    if audio_path is None:
        print("Audio extraction failed.")
        return False
    
    print("\n" + "="*50)
    print("STEP 3: Speech Recognition (Whisper)")
    print("="*50)
    transcript_path = transcribe_audio(audio_path)
    
    if transcript_path is None:
        print("Transcription failed.")
        return False
    
    print("\n" + "="*50)
    print("System Ready! Transcript saved to:", transcript_path)
    print("="*50)
    
    # Print transcript
    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            full_text = data.get("full_transcript", "")
            
            # Save raw txt
            txt_path = Path(transcript_path).with_suffix('.txt')
            with open(txt_path, 'w', encoding='utf-8') as txt_file:
                txt_file.write(full_text)
                
            print("\n--- TRANSCRIPT ---\n")
            # If the text is extremely long, just print the first 1000 chars to avoid console freeze
            if len(full_text) > 2000:
                print(full_text[:2000] + "\n\n... [TEXT TOO LONG TO PRINT COMPLETELY] ...")
            else:
                print(full_text)
            print("\n------------------\n")
            print(f"✅ Full raw text saved for easy copy-pasting at: {txt_path}")
    except Exception as e:
        print(f"Could not read transcript: {e}")
        
    return True

def main_menu():
    """Main menu loop."""
    while True:
        print("""
=================================================
MAIN MENU
=================================================

1 → Process New Video
2 → Exit
""")
        
        choice = input("Enter choice: ").strip()
        
        if choice == "1":
            process_pipeline()
        
        elif choice == "2":
            print("Exiting system.")
            sys.exit()
        
        else:
            print("Invalid choice.\n")

if __name__ == "__main__":
    setup_logger()
    print_header()
    main_menu()
