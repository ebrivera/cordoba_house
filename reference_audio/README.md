# Reference Audio Setup

To enable Naz voice detection, you need to provide reference audio samples of Naz speaking.

## Steps:

1. Find recordings where you know Naz is speaking (from known meetings)
2. Extract 10-30 second clips of clear Naz speech (no overlapping voices)
3. Save these as WAV, MP3, or M4A files in the reference_audio/ directory
4. Name them descriptively (e.g., naz_sample_1.wav, naz_sample_2.mp3)

## Tips:

- Use at least 3-5 different samples for best accuracy
- Ensure good audio quality (clear speech, minimal background noise)
- Longer samples (20-30 seconds) work better than very short ones
- You can use audio editing software to extract these clips

## File naming examples:
- naz_meeting_sample_1.wav
- naz_voice_reference_2.mp3
- naz_clear_speech_3.m4a

Once you have reference files, the Naz detection pipeline will automatically use them.
