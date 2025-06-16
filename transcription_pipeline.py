import os
import json
import time
from pathlib import Path
from datetime import datetime
import requests
import hashlib

class TranscriptionPipeline:
    def __init__(self, audio_dir="audio_files"):
        self.audio_dir = Path(audio_dir)
        self.metadata_dir = Path("metadata")
        self.transcripts_dir = Path("transcripts")
        
        # Create directories
        self.transcripts_dir.mkdir(exist_ok=True)
        self.metadata_dir.mkdir(exist_ok=True)
        
        # AssemblyAI setup
        self.api_key = os.getenv("ASSEMBLYAI_API_KEY")
        if not self.api_key:
            raise ValueError("ASSEMBLYAI_API_KEY not found in environment variables")
        
        self.upload_url = "https://api.assemblyai.com/v2/upload"
        self.transcript_url = "https://api.assemblyai.com/v2/transcript"
        self.headers = {"authorization": self.api_key}
        
        # Track processed transcriptions
        self.processed_file = self.metadata_dir / "processed_transcripts.json"
        self.processed_transcripts = self.load_processed_transcripts()
    
    def load_processed_transcripts(self):
        """Load list of already processed transcripts"""
        if self.processed_file.exists():
            with open(self.processed_file, 'r') as f:
                return set(json.load(f))
        return set()
    
    def save_processed_transcripts(self):
        """Save list of processed transcripts"""
        with open(self.processed_file, 'w') as f:
            json.dump(list(self.processed_transcripts), f)
    
    def create_transcript_id(self, audio_path):
        """Create unique ID for tracking processed transcripts"""
        audio_path = Path(audio_path)
        stat = audio_path.stat()
        content = f"{audio_path}_{stat.st_size}_{stat.st_mtime}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def upload_audio_file(self, audio_path):
        """Upload audio file to AssemblyAI and get upload URL"""
        try:
            print(f"📤 Uploading {audio_path.name} to AssemblyAI...")
            
            with open(audio_path, 'rb') as f:
                response = requests.post(self.upload_url, headers=self.headers, files={'file': f})
            
            if response.status_code == 200:
                upload_url = response.json()['upload_url']
                print(f"✅ Upload successful")
                return upload_url
            else:
                print(f"❌ Upload failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Upload error: {e}")
            return None
    
    def request_transcription(self, upload_url, enable_speaker_labels=True):
        """Request transcription with speaker diarization"""
        try:
            # Transcription configuration
            config = {
                "audio_url": upload_url,
                "speaker_labels": enable_speaker_labels,  # Enable speaker diarization
                "auto_highlights": True,  # Get key highlights
                "sentiment_analysis": False,  # Disable sentiment to save time
                "entity_detection": False,  # Disable entity detection to save time
                "auto_chapters": False,  # Disable chapters to save time
                "punctuate": True,  # Add punctuation
                "format_text": True,  # Format text nicely
            }
            
            print(f"🎯 Requesting transcription with speaker labels...")
            response = requests.post(self.transcript_url, json=config, headers=self.headers)
            
            if response.status_code == 200:
                transcript_id = response.json()['id']
                print(f"✅ Transcription requested - ID: {transcript_id}")
                return transcript_id
            else:
                print(f"❌ Transcription request failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Transcription request error: {e}")
            return None
    
    def wait_for_transcription(self, transcript_id, max_wait_minutes=30):
        """Wait for transcription to complete and return results"""
        print(f"⏳ Waiting for transcription to complete...")
        
        max_wait_seconds = max_wait_minutes * 60
        start_time = time.time()
        
        while True:
            # Check if we've exceeded max wait time
            elapsed = time.time() - start_time
            if elapsed > max_wait_seconds:
                print(f"⏰ Transcription timeout after {max_wait_minutes} minutes")
                return None
            
            try:
                # Get transcription status
                response = requests.get(f"{self.transcript_url}/{transcript_id}", headers=self.headers)
                
                if response.status_code == 200:
                    result = response.json()
                    status = result['status']
                    
                    if status == 'completed':
                        print(f"✅ Transcription completed!")
                        return result
                    elif status == 'error':
                        print(f"❌ Transcription failed: {result.get('error', 'Unknown error')}")
                        return None
                    else:
                        # Still processing
                        elapsed_min = elapsed / 60
                        print(f"   Status: {status} (elapsed: {elapsed_min:.1f}min)")
                        time.sleep(10)  # Wait 10 seconds before checking again
                else:
                    print(f"❌ Status check failed: {response.status_code}")
                    return None
                    
            except Exception as e:
                print(f"❌ Status check error: {e}")
                time.sleep(10)
    
    def save_transcript_results(self, transcript_data, audio_metadata):
        """Save transcription results to files"""
        # Create base filename from audio metadata
        audio_id = audio_metadata.get('audio_id', 'unknown')
        topic = audio_metadata.get('topic', 'unknown')
        start_time = audio_metadata.get('start_time', '')
        
        safe_topic = "".join(c for c in topic if c.isalnum() or c in (' ', '-', '_')).rstrip()[:30]
        date_str = start_time[:10] if start_time else 'unknown_date'
        
        base_filename = f"{date_str}_{safe_topic}_{audio_id}"
        
        # Save full transcript text
        transcript_file = self.transcripts_dir / f"{base_filename}_transcript.txt"
        with open(transcript_file, 'w', encoding='utf-8') as f:
            f.write(transcript_data['text'])
        
        # Save detailed transcript with speaker labels (if available)
        if transcript_data.get('utterances'):
            detailed_file = self.transcripts_dir / f"{base_filename}_detailed.txt"
            with open(detailed_file, 'w', encoding='utf-8') as f:
                f.write("=== DETAILED TRANSCRIPT WITH SPEAKER LABELS ===\n\n")
                
                for utterance in transcript_data['utterances']:
                    speaker = utterance.get('speaker', 'Unknown')
                    text = utterance.get('text', '')
                    start = utterance.get('start', 0) / 1000  # Convert to seconds
                    end = utterance.get('end', 0) / 1000
                    confidence = utterance.get('confidence', 0)
                    
                    f.write(f"[{start:.1f}s - {end:.1f}s] {speaker}: {text}\n")
                    f.write(f"    (Confidence: {confidence:.2f})\n\n")
        
        # Save raw JSON data
        json_file = self.transcripts_dir / f"{base_filename}_raw.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(transcript_data, f, indent=2)
        
        print(f"💾 Transcript saved:")
        print(f"   📝 Text: {transcript_file}")
        if transcript_data.get('utterances'):
            print(f"   🎤 Detailed: {detailed_file}")
        print(f"   📊 Raw data: {json_file}")
        
        return {
            "transcript_file": str(transcript_file),
            "detailed_file": str(detailed_file) if transcript_data.get('utterances') else None,
            "json_file": str(json_file)
        }
    
    def transcribe_audio_file(self, audio_path, audio_metadata):
        """Complete transcription workflow for a single audio file"""
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            print(f"❌ Audio file not found: {audio_path}")
            return None
        
        # Create unique ID for this transcription
        transcript_id = self.create_transcript_id(audio_path)
        
        # Skip if already processed
        if transcript_id in self.processed_transcripts:
            print(f"⏭️  Skipping already transcribed: {audio_path.name}")
            return None
        
        print(f"\n🎙️  Transcribing: {audio_path.name}")
        
        # Step 1: Upload audio file
        upload_url = self.upload_audio_file(audio_path)
        if not upload_url:
            return None
        
        # Step 2: Request transcription
        assemblyai_id = self.request_transcription(upload_url)
        if not assemblyai_id:
            return None
        
        # Step 3: Wait for completion
        transcript_data = self.wait_for_transcription(assemblyai_id)
        if not transcript_data:
            return None
        
        # Step 4: Save results
        file_paths = self.save_transcript_results(transcript_data, audio_metadata)
        
        # Mark as processed
        self.processed_transcripts.add(transcript_id)
        
        # Extract speaker information
        speakers_detected = []
        if transcript_data.get('utterances'):
            unique_speakers = set()
            for utterance in transcript_data['utterances']:
                speaker = utterance.get('speaker')
                if speaker:
                    unique_speakers.add(speaker)
            speakers_detected = list(unique_speakers)
        
        # Create transcription metadata
        transcription_metadata = {
            "transcript_id": transcript_id,
            "audio_file": str(audio_path),
            "assemblyai_id": assemblyai_id,
            "user_email": audio_metadata.get('user_email'),
            "meeting_id": audio_metadata.get('meeting_id'),
            "topic": audio_metadata.get('topic'),
            "start_time": audio_metadata.get('start_time'),
            "audio_duration_seconds": audio_metadata.get('audio_duration_seconds'),
            "speakers_detected": speakers_detected,
            "speaker_count": len(speakers_detected),
            "confidence": transcript_data.get('confidence', 0),
            "transcript_length": len(transcript_data.get('text', '')),
            "files": file_paths,
            "transcribed_at": datetime.now().isoformat()
        }
        
        return transcription_metadata
    
    def process_audio_metadata(self, audio_metadata_file):
        """Process all audio files from an audio metadata file"""
        audio_metadata_file = Path(audio_metadata_file)
        
        if not audio_metadata_file.exists():
            print(f"❌ Audio metadata file not found: {audio_metadata_file}")
            return []
        
        # Load audio metadata
        with open(audio_metadata_file, 'r') as f:
            audio_metadata_list = json.load(f)
        
        print(f"🎙️  Processing {len(audio_metadata_list)} audio files for transcription...")
        
        transcribed_files = []
        
        for i, audio_meta in enumerate(audio_metadata_list, 1):
            print(f"\n📊 Progress: {i}/{len(audio_metadata_list)}")
            print(f"🎵 File: {audio_meta.get('topic', 'Unknown')} - {audio_meta.get('recording_type', 'Unknown')}")
            
            transcript_meta = self.transcribe_audio_file(audio_meta['audio_file'], audio_meta)
            
            if transcript_meta:
                transcribed_files.append(transcript_meta)
        
        # Save processed transcripts list
        self.save_processed_transcripts()
        
        # Save transcription metadata
        transcript_metadata_file = self.metadata_dir / f"transcription_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(transcript_metadata_file, 'w') as f:
            json.dump(transcribed_files, f, indent=2)
        
        print(f"\n✅ Transcription processing complete!")
        print(f"📊 Total files transcribed: {len(transcribed_files)}")
        print(f"📝 Transcripts saved in: {self.transcripts_dir}")
        print(f"📋 Transcription metadata saved to: {transcript_metadata_file}")
        
        return transcribed_files
    
    def process_latest_audio(self):
        """Find and process the most recent audio metadata file"""
        metadata_files = list(self.metadata_dir.glob("audio_metadata_*.json"))
        
        if not metadata_files:
            print("❌ No audio metadata files found. Run the audio processor first.")
            return []
        
        # Get the most recent metadata file
        latest_file = max(metadata_files, key=lambda p: p.stat().st_mtime)
        print(f"📂 Using latest audio metadata: {latest_file.name}")
        
        return self.process_audio_metadata(latest_file)

def main():
    # Check for AssemblyAI API key
    if not os.getenv("ASSEMBLYAI_API_KEY"):
        print("❌ ASSEMBLYAI_API_KEY not found in environment variables")
        print("   Add it to your .env file: ASSEMBLYAI_API_KEY=your_key_here")
        return
    
    # Initialize transcription pipeline
    pipeline = TranscriptionPipeline()
    
    # Process the latest audio files
    transcription_metadata = pipeline.process_latest_audio()
    
    if transcription_metadata:
        print(f"\n🎉 Transcription pipeline complete!")
        print(f"Transcribed {len(transcription_metadata)} audio files")
        
        # Summary of speaker detection
        multi_speaker_count = sum(1 for t in transcription_metadata if t['speaker_count'] > 1)
        print(f"🎤 Files with multiple speakers detected: {multi_speaker_count}")
        print(f"   (These are likely meetings rather than solo lectures)")
    else:
        print("❌ No audio files were transcribed")

if __name__ == "__main__":
    main()