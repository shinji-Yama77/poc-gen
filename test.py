from speaker_diarizer import SpeakerDiarizer

# Create diarizer
diarizer = SpeakerDiarizer(whisper_model="base")

# Get word-level results
word_results = diarizer.transcribe_with_speakers("recordings/your_conversation.wav")

# Format into speaker segments
segments = diarizer.format_transcript(word_results)

# Save files
diarizer.save_transcript(segments, "my_transcript.txt")