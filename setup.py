#!/usr/bin/env python3
"""
Setup script for Zoom Recording Pipeline
Creates directories, checks dependencies, and helps configure environment
"""

import os
import sys
from pathlib import Path
import subprocess

def create_directories():
    """Create necessary directories for the pipeline"""
    directories = [
        "downloads",
        "audio_files", 
        "transcripts",
        "metadata",
        "classification_results",
        "reference_audio"
    ]
    
    print("📁 Creating directories...")
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"   ✅ {directory}/")
    
    print()

def check_ffmpeg():
    """Check if FFmpeg is installed"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ FFmpeg is installed")
            return True
        else:
            print("❌ FFmpeg not found")
            return False
    except FileNotFoundError:
        print("❌ FFmpeg not found")
        return False

def create_env_template():
    """Create .env template file"""
    env_template = """# Zoom API Credentials
ZOOM_CLIENT_ID=your_zoom_client_id_here
ZOOM_CLIENT_SECRET=your_zoom_client_secret_here
ZOOM_ACCOUNT_ID=your_zoom_account_id_here

# AssemblyAI API Key (for transcription)
ASSEMBLYAI_API_KEY=your_assemblyai_api_key_here

# Optional: OpenAI API Key (if using OpenAI Whisper API instead of local)
# OPENAI_API_KEY=your_openai_api_key_here
"""
    
    env_file = Path(".env")
    if not env_file.exists():
        with open(env_file, 'w') as f:
            f.write(env_template)
        print("📄 Created .env template file")
        print("   ⚠️  Please edit .env with your actual API credentials")
    else:
        print("📄 .env file already exists")

def install_requirements():
    """Install Python requirements"""
    try:
        print("📦 Installing Python requirements...")
        result = subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'],
                              check=True)
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements")
        return False

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - requires Python 3.8+")
        return False

def create_reference_audio_instructions():
    """Create instructions for reference audio"""
    instructions = """# Reference Audio Setup

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
"""
    
    instructions_file = Path("reference_audio/README.md")
    with open(instructions_file, 'w') as f:
        f.write(instructions)
    
    print("📋 Created reference audio instructions: reference_audio/README.md")

def main():
    print("🚀 Zoom Recording Pipeline Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        print("\n❌ Setup failed: Unsupported Python version")
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Create .env template
    create_env_template()
    
    # Create reference audio instructions
    create_reference_audio_instructions()
    
    # Install requirements
    if Path("requirements.txt").exists():
        install_success = install_requirements()
    else:
        print("⚠️  requirements.txt not found - skipping package installation")
        install_success = True
    
    # Check FFmpeg
    print("\n🔧 Checking system dependencies...")
    ffmpeg_ok = check_ffmpeg()
    
    # Summary
    print("\n" + "=" * 40)
    print("📋 Setup Summary:")
    print(f"   📁 Directories: ✅ Created")
    print(f"   📄 Environment: ✅ Template created")
    print(f"   📦 Python packages: {'✅ Installed' if install_success else '❌ Failed'}")
    print(f"   🔧 FFmpeg: {'✅ Available' if ffmpeg_ok else '❌ Missing'}")
    
    if not ffmpeg_ok:
        print("\n⚠️  FFmpeg is required for audio processing")
        print("   Install with:")
        print("   - macOS: brew install ffmpeg")
        print("   - Ubuntu/Debian: sudo apt-get install ffmpeg")
        print("   - Windows: Download from https://ffmpeg.org/download.html")
    
    print("\n📝 Next steps:")
    print("1. Edit .env file with your API credentials")
    print("2. Add Naz's voice samples to reference_audio/ directory")
    if not ffmpeg_ok:
        print("3. Install FFmpeg")
    print(f"4. Run the pipeline: streamlit run streamlit_dashboard.py")
    
    print("\n🎉 Setup complete!")

if __name__ == "__main__":
    main()