#!/usr/bin/env python3
"""
Complete Pipeline Runner
Runs the entire Zoom recording processing pipeline
"""

import sys
import argparse
from pathlib import Path

# Import all pipeline modules
from zoom_pipeline import ZoomDownloader
from audio_processor import AudioProcessor  
from transcription_pipeline import TranscriptionPipeline
from naz_detection import NazDetectionPipeline

def run_full_pipeline(user_emails, skip_existing=True):
    """Run the complete pipeline"""
    
    print("🚀 Starting Complete Zoom Recording Pipeline")
    print("=" * 60)
    
    try:
        # Step 1: Download recordings
        print("\n1️⃣ DOWNLOADING RECORDINGS")
        print("-" * 30)
        downloader = ZoomDownloader()
        downloaded_files = downloader.process_recordings(user_emails)
        
        if not downloaded_files:
            print("❌ No files downloaded. Pipeline stopped.")
            return False
        
        print(f"✅ Downloaded {len(downloaded_files)} files")
        
        # Step 2: Process audio
        print("\n2️⃣ PROCESSING AUDIO")
        print("-" * 30)
        audio_processor = AudioProcessor()
        audio_metadata = audio_processor.process_latest_downloads()
        
        if not audio_metadata:
            print("❌ No audio files processed. Pipeline stopped.")
            return False
        
        print(f"✅ Processed {len(audio_metadata)} audio files")
        
        # Step 3: Transcribe audio
        print("\n3️⃣ TRANSCRIBING AUDIO")
        print("-" * 30)
        transcription_pipeline = TranscriptionPipeline()
        transcription_metadata = transcription_pipeline.process_latest_audio()
        
        if not transcription_metadata:
            print("❌ No transcriptions created. Pipeline stopped.")
            return False
        
        print(f"✅ Transcribed {len(transcription_metadata)} files")
        
        # Step 4: Detect Naz voice
        print("\n4️⃣ DETECTING NAZ VOICE")
        print("-" * 30)
        naz_detection = NazDetectionPipeline()
        detection_results = naz_detection.process_latest_transcriptions()
        
        if not detection_results:
            print("❌ No voice detection results. Pipeline stopped.")
            return False
        
        # Summary
        meetings = [r for r in detection_results if r['classification'] == 'meeting']
        lectures = [r for r in detection_results if r['classification'] == 'lecture']
        
        print(f"✅ Analyzed {len(detection_results)} recordings")
        print(f"   🤝 Meetings with Naz: {len(meetings)}")
        print(f"   🎓 Solo lectures: {len(lectures)}")
        
        print("\n🎉 PIPELINE COMPLETE!")
        print("=" * 60)
        print("📊 Results Summary:")
        print(f"   • Downloaded: {len(downloaded_files)} files")
        print(f"   • Audio processed: {len(audio_metadata)} files")
        print(f"   • Transcribed: {len(transcription_metadata)} files")
        print(f"   • Classified: {len(detection_results)} recordings")
        print(f"   • Meetings found: {len(meetings)}")
        print(f"   • Lectures found: {len(lectures)}")
        
        if meetings:
            print("\n📅 Meetings with Naz detected:")
            for meeting in meetings:
                naz_time = meeting.get('naz_speaking_time_seconds', 0) / 60
                print(f"   • {meeting['topic']} ({meeting['start_time'][:10]}) - {naz_time:.1f}min Naz")
        
        print(f"\n📁 Check these directories for results:")
        print(f"   • Downloads: downloads/")
        print(f"   • Audio files: audio_files/")
        print(f"   • Transcripts: transcripts/")
        print(f"   • Results: classification_results/")
        print(f"   • Metadata: metadata/")
        
        return True
        
    except KeyboardInterrupt:
        print("\n⚠️ Pipeline interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_individual_step(step, user_emails=None):
    """Run an individual pipeline step"""
    
    if step == "download":
        print("🔽 Running: Download Recordings")
        if not user_emails:
            print("❌ User emails required for download step")
            return False
        downloader = ZoomDownloader()
        result = downloader.process_recordings(user_emails)
        return len(result) > 0 if result else False
        
    elif step == "audio":
        print("🎵 Running: Audio Processing")
        processor = AudioProcessor()
        result = processor.process_latest_downloads()
        return len(result) > 0 if result else False
        
    elif step == "transcribe":
        print("📝 Running: Transcription")
        pipeline = TranscriptionPipeline()
        result = pipeline.process_latest_audio()
        return len(result) > 0 if result else False
        
    elif step == "detect":
        print("🎯 Running: Naz Detection")
        detector = NazDetectionPipeline()
        result = detector.process_latest_transcriptions()
        return len(result) > 0 if result else False
        
    else:
        print(f"❌ Unknown step: {step}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Zoom Recording Processing Pipeline")
    
    # Mode selection
    parser.add_argument(
        "--mode", 
        choices=["full", "step"], 
        default="full",
        help="Run full pipeline or individual step"
    )
    
    # Individual step selection
    parser.add_argument(
        "--step",
        choices=["download", "audio", "transcribe", "detect"],
        help="Individual step to run (requires --mode step)"
    )
    
    # User emails
    parser.add_argument(
        "--emails",
        nargs="+",
        default=["cordobahouseschool3@gmail.com"],
        help="Zoom user emails to process"
    )
    
    # Options
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip already processed files"
    )
    
    # Dashboard
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Launch Streamlit dashboard instead of running pipeline"
    )
    
    args = parser.parse_args()
    
    # Launch dashboard
    if args.dashboard:
        print("🚀 Launching Streamlit Dashboard...")
        import subprocess
        subprocess.run(["streamlit", "run", "streamlit_dashboard.py"])
        return
    
    # Validate arguments
    if args.mode == "step" and not args.step:
        print("❌ --step required when using --mode step")
        parser.print_help()
        sys.exit(1)
    
    # Check environment setup
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ .env file not found")
        print("   Run: python setup.py")
        sys.exit(1)
    
    # Run pipeline
    if args.mode == "full":
        success = run_full_pipeline(args.emails, args.skip_existing)
    else:
        success = run_individual_step(args.step, args.emails if args.step == "download" else None)
    
    if success:
        print("\n✅ Execution completed successfully")
        if args.mode == "full":
            print("\n🌐 To view results in dashboard:")
            print("   python run_pipeline.py --dashboard")
    else:
        print("\n❌ Execution failed")
        sys.exit(1)

if __name__ == "__main__":
    main()