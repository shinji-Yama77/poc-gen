from speaker_diarizer import SpeakerDiarizer
from pathlib import Path
from typing import Optional

def diarize_audio(audio_file_path: str, hf_token: Optional[str] = None, whisper_model: str = "base") -> dict:
    """
    Simple function to run speaker diarization with transcription.
    
    Args:
        audio_file_path: Path to the audio file (WAV, MP3, etc.)
        hf_token: Hugging Face token. If None, reads from .env file
        whisper_model: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
    
    Returns:
        dict: Paths to generated transcript and SRT files, plus segments data
    """
    try:
        # Initialize the diarizer
        diarizer = SpeakerDiarizer(hf_token=hf_token, whisper_model=whisper_model)
        
        # Process and save
        result = diarizer.process_and_save(audio_file_path)
        
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise

def main():
    print("Hello from poc-gen!")

if __name__ == "__main__":
    main()
