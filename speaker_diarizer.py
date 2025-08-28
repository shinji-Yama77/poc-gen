import os
import torch
import whisper
from dotenv import load_dotenv
from pyannote.audio import Pipeline
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

class SpeakerDiarizer:
    """
    A class for performing speaker diarization with word-level transcription.
    
    This class combines speaker identification with word-level speech transcription
    to provide precise speaker-aware transcripts of audio files.
    """
    
    def __init__(self, hf_token: Optional[str] = None, whisper_model: str = "base"):
        """
        Initialize the SpeakerDiarizer.
        
        Args:
            hf_token: Hugging Face token for pyannote.audio. If None, reads from .env
            whisper_model: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
        """
        # Load environment variables
        load_dotenv()
        
        # Get Hugging Face token
        self.hf_token = hf_token or os.getenv("HUGGINGFACE_ACCESS_TOKEN")
        if not self.hf_token:
            raise ValueError("HUGGINGFACE_ACCESS_TOKEN not found. Set it in .env file or pass as parameter.")
        
        # Initialize models
        self.whisper_model_name = whisper_model
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize the diarization and transcription models."""
        print("🔧 Initializing models...")
        
        # Initialize Whisper model
        self.whisper_model = whisper.load_model(self.whisper_model_name)
        
        # Initialize speaker diarization pipeline
        self.diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=self.hf_token
        )
        
        # Send to appropriate device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.diarization_pipeline.to(self.device)
        
        print(f"✅ Models initialized on {self.device}")
    
    def transcribe_with_speakers(self, audio_file_path: str) -> List[Dict]:
        """
        Transcribe audio with word-level timestamps and speaker identification.
        
        Args:
            audio_file_path: Path to the audio file
            
        Returns:
            List of dictionaries containing word-level speaker segments
        """
        # Validate input file
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
        
        print(f"🔍 Analyzing speakers in: {audio_file_path}")
        
        # Get transcription with word-level timestamps
        print("📝 Transcribing with word-level timestamps...")
        transcription = self.whisper_model.transcribe(
            audio_file_path, 
            word_timestamps=True  # This is key!
        )
        
        # Get speaker diarization
        print("🎤 Identifying speakers...")
        diarization = self.diarization_pipeline(audio_file_path)
        
        # Combine transcription with speaker information
        print("🔗 Combining transcription with speaker data...")
        word_results = []
        
        # Go through each word in transcription
        for segment in transcription["segments"]:
            for word_info in segment.get("words", []):
                word_start = word_info["start"]
                word_end = word_info["end"]
                word_text = word_info["word"]
                
                # Find which speaker was talking at this time
                speaker = self._find_speaker_at_time(diarization, word_start)
                
                word_results.append({
                    "start": word_start,
                    "end": word_end,
                    "text": word_text,
                    "speaker": speaker
                })
        
        return word_results
    
    def _find_speaker_at_time(self, diarization, timestamp: float) -> str:
        """
        Find which speaker was active at a given timestamp.
        
        Args:
            diarization: Diarization result from pyannote.audio
            timestamp: Time in seconds
            
        Returns:
            Speaker identifier or "UNKNOWN"
        """
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if turn.start <= timestamp <= turn.end:
                return f"Speaker_{speaker}"
        return "UNKNOWN"
    
    def format_transcript(self, word_results: List[Dict]) -> List[Dict]:
        """
        Group words by speaker and create readable transcript.
        
        Args:
            word_results: List of word-level results from transcribe_with_speakers()
            
        Returns:
            List of speaker segments with combined text
        """
        if not word_results:
            return []
        
        transcript = []
        current_speaker = word_results[0]["speaker"]
        current_text = ""
        current_start = word_results[0]["start"]
        
        for word_data in word_results:
            if word_data["speaker"] == current_speaker:
                current_text += word_data["text"]
            else:
                # Speaker changed, save previous segment
                if current_text.strip():  # Only add non-empty segments
                    transcript.append({
                        "start": current_start,
                        "end": word_data["start"],
                        "speaker": current_speaker,
                        "text": current_text.strip()
                    })
                
                # Start new segment
                current_speaker = word_data["speaker"]
                current_text = word_data["text"]
                current_start = word_data["start"]
        
        # Add final segment
        if current_text.strip():
            transcript.append({
                "start": current_start,
                "end": word_results[-1]["end"],
                "speaker": current_speaker,
                "text": current_text.strip()
            })
        
        return transcript
    
    def process_audio(self, audio_file_path: str) -> List[Dict]:
        """
        Process an audio file to get speaker diarization with transcriptions.
        
        Args:
            audio_file_path: Path to the audio file
            
        Returns:
            List of dictionaries containing speaker segments with transcriptions
        """
        # Get word-level results
        word_results = self.transcribe_with_speakers(audio_file_path)
        
        # Format into speaker segments
        speaker_segments = self.format_transcript(word_results)
        
        # Print results
        for segment in speaker_segments:
            print(f"🎤 [{segment['start']:.1f}s-{segment['end']:.1f}s] "
                  f"{segment['speaker']}: {segment['text']}")
        
        return speaker_segments
    
    def save_transcript(self, speaker_segments: List[Dict], output_file: str = "transcript.txt"):
        """
        Save the transcript to a text file.
        
        Args:
            speaker_segments: List of speaker segments from process_audio()
            output_file: Path for the output file
        """
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("Speaker Diarization Transcript\n")
            f.write("=" * 40 + "\n\n")
            
            for segment in speaker_segments:
                f.write(f"{segment['speaker']} ({segment['start']:.1f}s - {segment['end']:.1f}s):\n")
                f.write(f"{segment['text']}\n\n")
        
        print(f"💾 Transcript saved to: {output_file}")
    
    def save_srt(self, speaker_segments: List[Dict], output_file: str = "transcript.srt"):
        """
        Save the transcript as an SRT subtitle file.
        
        Args:
            speaker_segments: List of speaker segments from process_audio()
            output_file: Path for the output file
        """
        with open(output_file, "w", encoding="utf-8") as f:
            for i, segment in enumerate(speaker_segments, 1):
                # Convert timestamps to SRT format (HH:MM:SS,mmm)
                start_srt = datetime.utcfromtimestamp(segment['start']).strftime('%H:%M:%S,%f')[:-3]
                end_srt = datetime.utcfromtimestamp(segment['end']).strftime('%H:%M:%S,%f')[:-3]
                
                f.write(f"{i}\n")
                f.write(f"{start_srt} --> {end_srt}\n")
                f.write(f"{segment['speaker']}: {segment['text']}\n\n")
        
        print(f"🎬 SRT file saved to: {output_file}")
    
    def process_and_save(self, audio_file_path: str, output_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Process audio and save both transcript and SRT files.
        
        Args:
            audio_file_path: Path to the audio file
            output_dir: Directory for output files (defaults to audio file directory)
            
        Returns:
            Dictionary with paths to generated files
        """
        # Process the audio
        segments = self.process_audio(audio_file_path)
        
        # Determine output directory
        if output_dir is None:
            output_dir = Path(audio_file_path).parent
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
        
        # Generate output file paths
        base_name = Path(audio_file_path).stem
        transcript_file = output_dir / f"{base_name}_transcript.txt"
        srt_file = output_dir / f"{base_name}_transcript.srt"
        
        # Save files
        self.save_transcript(segments, str(transcript_file))
        self.save_srt(segments, str(srt_file))
        
        print(f"\n✅ Processing complete! Found {len(segments)} speaker segments.")
        
        return {
            "transcript": str(transcript_file),
            "srt": str(srt_file),
            "audio_file": audio_file_path,
            "segments": segments
        }


# Example usage
if __name__ == "__main__":
    try:
        # Initialize the diarizer
        diarizer = SpeakerDiarizer(whisper_model="base")
        
        # Process an audio file
        audio_file = "recordings/your_conversation.wav"  # Change this to your audio file path
        result = diarizer.process_and_save(audio_file)
        
        print(f"📄 Transcript: {result['transcript']}")
        print(f"🎬 SRT: {result['srt']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you have:")
        print("1. HUGGINGFACE_ACCESS_TOKEN in your .env file")
        print("2. A valid audio file path")
        print("3. Required packages installed")