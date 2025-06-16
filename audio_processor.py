import os
import json
from pathlib import Path
from pydub import AudioSegment
import subprocess
import hashlib
from datetime import datetime

class AudioProcessor:
    def __init__(self, downloads_dir="downloads", audio_dir="audio_files"):
        self.downloads_dir = Path(downloads_dir)
        self.audio_dir = Path(audio_dir)
        self.metadata_dir = Path("metadata")
        
        # Create directories
        self.audio_dir.mkdir(exist_ok=True)
        self.metadata_dir.mkdir(exist_ok=True)
        
        # Track processed files
        self.processed_file = self.metadata_dir / "processed_audio.json"
        self.processed_audio = self.load_processed_audio()
        
        # Supported video formats that need audio extraction
        self.video_formats = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        # Audio formats that can be used directly or converted
        self.audio_formats = {'.mp3', '.m4a', '.wav', '.aac', '.flac', '.ogg'}
    
    def load_processed_audio(self):
        """Load list of already processed audio files"""
        if self.processed_file.exists():
            with open(self.processed_file, 'r') as f:
                return set(json.load(f))
        return set()
    
    def save_processed_audio(self):
        """Save list of processed audio files"""
        with open(self.processed_file, 'w') as f:
            json.dump(list(self.processed_audio), f)
    
    def get_audio_duration(self, file_path):
        """Get duration of audio/video file using ffprobe"""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json', 
                '-show_format', str(file_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            return float(data['format']['duration'])
        except:
            return None
    
    def extract_audio_from_video(self, video_path, audio_path):
        """Extract audio from video file using ffmpeg"""
        try:
            cmd = [
                'ffmpeg', '-i', str(video_path),
                '-vn',  # No video
                '-acodec', 'libmp3lame',  # Use MP3 codec
                '-ar', '16000',  # 16kHz sample rate (good for speech)
                '-ac', '1',  # Mono
                '-ab', '128k',  # 128kbps bitrate
                '-y',  # Overwrite output file
                str(audio_path)
            ]
            
            print(f"🎵 Extracting audio: {video_path.name} -> {audio_path.name}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            if audio_path.exists():
                print(f"✅ Audio extracted successfully")
                return True
            else:
                print(f"❌ Audio extraction failed - file not created")
                return False
                
        except subprocess.CalledProcessError as e:
            print(f"❌ FFmpeg error: {e}")
            print(f"   stderr: {e.stderr}")
            return False
        except Exception as e:
            print(f"❌ Extraction error: {e}")
            return False
    
    def convert_audio_format(self, input_path, output_path):
        """Convert audio to standardized format using pydub"""
        try:
            print(f"🔄 Converting audio: {input_path.name} -> {output_path.name}")
            
            # Load audio file
            audio = AudioSegment.from_file(str(input_path))
            
            # Standardize: 16kHz, mono, MP3
            audio = audio.set_frame_rate(16000)
            audio = audio.set_channels(1)  # Convert to mono
            
            # Export as MP3
            audio.export(str(output_path), format="mp3", bitrate="128k")
            
            if output_path.exists():
                print(f"✅ Audio converted successfully")
                return True
            else:
                print(f"❌ Audio conversion failed")
                return False
                
        except Exception as e:
            print(f"❌ Audio conversion error: {e}")
            return False
    
    def create_audio_id(self, file_path):
        """Create unique ID for tracking processed files"""
        # Use file path and modification time for uniqueness
        stat = file_path.stat()
        content = f"{file_path}_{stat.st_size}_{stat.st_mtime}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def process_file(self, file_path, metadata):
        """Process a single recording file to extract/convert audio"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            print(f"⚠️  File not found: {file_path}")
            return None
        
        # Create unique ID for this file
        file_id = self.create_audio_id(file_path)
        
        # Skip if already processed
        if file_id in self.processed_audio:
            print(f"⏭️  Skipping already processed: {file_path.name}")
            return None
        
        # Determine output filename
        meeting_id = metadata.get('meeting_id', 'unknown')
        recording_type = metadata.get('recording_type', 'unknown')
        topic = metadata.get('topic', 'unknown')
        start_time = metadata.get('start_time', '')
        
        # Clean topic for filename
        safe_topic = "".join(c for c in topic if c.isalnum() or c in (' ', '-', '_')).rstrip()[:30]
        date_str = start_time[:10] if start_time else 'unknown_date'
        
        output_filename = f"{date_str}_{safe_topic}_{recording_type}_{meeting_id}.mp3"
        output_path = self.audio_dir / output_filename
        
        # Get file extension
        file_ext = file_path.suffix.lower()
        
        success = False
        
        if file_ext in self.video_formats:
            # Extract audio from video
            success = self.extract_audio_from_video(file_path, output_path)
            
        elif file_ext in self.audio_formats:
            # Convert audio to standard format
            success = self.convert_audio_format(file_path, output_path)
            
        else:
            print(f"⚠️  Unsupported file format: {file_ext}")
            return None
        
        if success:
            # Mark as processed
            self.processed_audio.add(file_id)
            
            # Get audio duration
            duration = self.get_audio_duration(output_path)
            
            # Create audio metadata
            audio_metadata = {
                "audio_id": file_id,
                "original_file": str(file_path),
                "audio_file": str(output_path),
                "user_email": metadata.get('user_email'),
                "meeting_id": meeting_id,
                "topic": topic,
                "start_time": start_time,
                "duration_minutes": metadata.get('duration'),
                "audio_duration_seconds": duration,
                "recording_type": recording_type,
                "file_type": metadata.get('file_type'),
                "original_size": metadata.get('file_size'),
                "audio_size": output_path.stat().st_size if output_path.exists() else None,
                "processed_at": datetime.now().isoformat()
            }
            
            return audio_metadata
        
        return None
    
    def process_downloads_metadata(self, metadata_file):
        """Process all files from a downloads metadata file"""
        metadata_file = Path(metadata_file)
        
        if not metadata_file.exists():
            print(f"❌ Metadata file not found: {metadata_file}")
            return []
        
        # Load download metadata
        with open(metadata_file, 'r') as f:
            download_metadata = json.load(f)
        
        print(f"🎵 Processing {len(download_metadata)} downloaded files for audio extraction...")
        
        processed_audio = []
        
        for file_meta in download_metadata:
            print(f"\n📁 Processing: {file_meta.get('topic', 'Unknown')} - {file_meta.get('recording_type', 'Unknown')}")
            
            audio_meta = self.process_file(file_meta['local_path'], file_meta)
            
            if audio_meta:
                processed_audio.append(audio_meta)
        
        # Save processed audio metadata
        self.save_processed_audio()
        
        audio_metadata_file = self.metadata_dir / f"audio_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(audio_metadata_file, 'w') as f:
            json.dump(processed_audio, f, indent=2)
        
        print(f"\n✅ Audio processing complete!")
        print(f"📊 Total audio files created: {len(processed_audio)}")
        print(f"🎵 Audio files saved in: {self.audio_dir}")
        print(f"📝 Audio metadata saved to: {audio_metadata_file}")
        
        return processed_audio
    
    def process_latest_downloads(self):
        """Find and process the most recent downloads metadata file"""
        metadata_files = list(self.metadata_dir.glob("downloads_*.json"))
        
        if not metadata_files:
            print("❌ No download metadata files found. Run the downloader first.")
            return []
        
        # Get the most recent metadata file
        latest_file = max(metadata_files, key=lambda p: p.stat().st_mtime)
        print(f"📂 Using latest downloads metadata: {latest_file.name}")
        
        return self.process_downloads_metadata(latest_file)

def main():
    # Initialize audio processor
    processor = AudioProcessor()
    
    # Process the latest downloads
    audio_metadata = processor.process_latest_downloads()
    
    if audio_metadata:
        print(f"\n🎉 Audio processing complete!")
        print(f"Created {len(audio_metadata)} audio files ready for transcription and diarization")
    else:
        print("❌ No audio files were processed")

if __name__ == "__main__":
    main()