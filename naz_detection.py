import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import torch
from speechbrain.pretrained import EncoderClassifier
from pydub import AudioSegment
import tempfile
import librosa
from sklearn.metrics.pairwise import cosine_similarity

class NazDetectionPipeline:
    def __init__(self, audio_dir="audio_files", reference_audio_dir="reference_audio"):
        self.audio_dir = Path(audio_dir)
        self.reference_audio_dir = Path(reference_audio_dir)
        self.metadata_dir = Path("metadata")
        self.results_dir = Path("classification_results")
        
        # Create directories
        self.reference_audio_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize speaker embedding model
        print("🧠 Loading speaker embedding model...")
        self.embedding_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb"
        )
        print("✅ Speaker embedding model loaded")
        
        # Track processed files
        self.processed_file = self.metadata_dir / "processed_naz_detection.json"
        self.processed_detections = self.load_processed_detections()
        
        # Store Naz's reference embedding
        self.naz_embedding = None
        self.similarity_threshold = 0.75  # Adjustable threshold for Naz detection
    
    def load_processed_detections(self):
        """Load list of already processed detections"""
        if self.processed_file.exists():
            with open(self.processed_file, 'r') as f:
                return set(json.load(f))
        return set()
    
    def save_processed_detections(self):
        """Save list of processed detections"""
        with open(self.processed_file, 'w') as f:
            json.dump(list(self.processed_detections), f)
    
    def extract_audio_segment(self, audio_file, start_time, end_time):
        """Extract a segment from audio file"""
        try:
            # Load audio
            audio = AudioSegment.from_file(audio_file)
            
            # Extract segment (times in seconds, convert to milliseconds)
            start_ms = int(start_time * 1000)
            end_ms = int(end_time * 1000)
            segment = audio[start_ms:end_ms]
            
            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            segment.export(temp_file.name, format="wav")
            
            return temp_file.name
        except Exception as e:
            print(f"❌ Error extracting audio segment: {e}")
            return None
    
    def get_speaker_embedding(self, audio_file, duration_limit=30):
        """Get speaker embedding from audio file"""
        try:
            # Load audio with librosa (SpeechBrain expects specific format)
            audio_data, sample_rate = librosa.load(audio_file, sr=16000)
            
            # Limit duration to avoid memory issues and improve speed
            if duration_limit and len(audio_data) > duration_limit * sample_rate:
                audio_data = audio_data[:duration_limit * sample_rate]
            
            # Convert to tensor
            audio_tensor = torch.tensor(audio_data).unsqueeze(0)
            
            # Get embedding
            with torch.no_grad():
                embedding = self.embedding_model.encode_batch(audio_tensor)
                # Convert to numpy array
                embedding = embedding.squeeze().cpu().numpy()
            
            return embedding
            
        except Exception as e:
            print(f"❌ Error getting speaker embedding: {e}")
            return None
    
    def create_naz_reference_embedding(self, reference_files=None):
        """Create or load Naz's reference embedding"""
        reference_embedding_file = self.metadata_dir / "naz_reference_embedding.npy"
        
        # Try to load existing embedding
        if reference_embedding_file.exists():
            try:
                self.naz_embedding = np.load(reference_embedding_file)
                print(f"✅ Loaded existing Naz reference embedding")
                return True
            except:
                print("⚠️  Failed to load existing embedding, will recreate")
        
        # Create new embedding from reference files
        if reference_files is None:
            # Look for reference files in reference directory
            reference_files = list(self.reference_audio_dir.glob("*.wav")) + \
                            list(self.reference_audio_dir.glob("*.mp3")) + \
                            list(self.reference_audio_dir.glob("*.m4a"))
        
        if not reference_files:
            print(f"❌ No reference audio files found in {self.reference_audio_dir}")
            print(f"   Please add Naz's voice samples to this directory")
            return False
        
        print(f"🎯 Creating Naz reference embedding from {len(reference_files)} files...")
        
        embeddings = []
        for ref_file in reference_files:
            print(f"   Processing: {ref_file.name}")
            embedding = self.get_speaker_embedding(ref_file)
            if embedding is not None:
                embeddings.append(embedding)
        
        if not embeddings:
            print("❌ Failed to create any embeddings from reference files")
            return False
        
        # Average multiple embeddings for robustness
        self.naz_embedding = np.mean(embeddings, axis=0)
        
        # Save the reference embedding
        np.save(reference_embedding_file, self.naz_embedding)
        print(f"✅ Naz reference embedding created and saved")
        
        return True
    
    def detect_naz_in_segments(self, transcript_data, audio_file):
        """Detect if Naz is speaking in each segment of the transcript"""
        if not transcript_data.get('utterances'):
            print("⚠️  No speaker segments found in transcript")
            return []
        
        if self.naz_embedding is None:
            print("❌ Naz reference embedding not available")
            return []
        
        print(f"🔍 Analyzing {len(transcript_data['utterances'])} speaker segments...")
        
        segment_results = []
        speaker_embeddings = {}  # Cache embeddings for each speaker
        
        for i, utterance in enumerate(transcript_data['utterances']):
            speaker = utterance.get('speaker', 'Unknown')
            start_time = utterance.get('start', 0) / 1000  # Convert to seconds
            end_time = utterance.get('end', 0) / 1000
            text = utterance.get('text', '')
            confidence = utterance.get('confidence', 0)
            
            print(f"   Segment {i+1}/{len(transcript_data['utterances'])}: {speaker} ({start_time:.1f}s-{end_time:.1f}s)")
            
            # Skip very short segments (less than 2 seconds)
            if end_time - start_time < 2:
                print(f"     Skipping short segment ({end_time - start_time:.1f}s)")
                continue
            
            # Check if we already have embedding for this speaker
            if speaker not in speaker_embeddings:
                # Extract audio segment
                temp_audio_file = self.extract_audio_segment(audio_file, start_time, end_time)
                
                if temp_audio_file:
                    # Get embedding for this segment
                    segment_embedding = self.get_speaker_embedding(temp_audio_file)
                    
                    # Clean up temporary file
                    try:
                        os.unlink(temp_audio_file)
                    except:
                        pass
                    
                    if segment_embedding is not None:
                        speaker_embeddings[speaker] = segment_embedding
                    else:
                        print(f"     Failed to get embedding for {speaker}")
                        continue
                else:
                    print(f"     Failed to extract audio segment for {speaker}")
                    continue
            
            # Calculate similarity with Naz's reference
            speaker_embedding = speaker_embeddings[speaker]
            similarity = cosine_similarity(
                self.naz_embedding.reshape(1, -1),
                speaker_embedding.reshape(1, -1)
            )[0][0]
            
            is_naz = similarity > self.similarity_threshold
            
            print(f"     Similarity to Naz: {similarity:.3f} ({'✅ NAZ' if is_naz else '❌ Not Naz'})")
            
            segment_result = {
                "segment_index": i,
                "speaker": speaker,
                "start_time": start_time,
                "end_time": end_time,
                "duration": end_time - start_time,
                "text": text,
                "confidence": confidence,
                "similarity_to_naz": float(similarity),
                "is_naz": is_naz
            }
            
            segment_results.append(segment_result)
        
        return segment_results
    
    def classify_recording(self, segment_results):
        """Classify recording as meeting or lecture based on Naz detection"""
        if not segment_results:
            return {
                "classification": "lecture",
                "confidence": "low",
                "reason": "No speaker segments analyzed",
                "naz_segments": 0,
                "total_segments": 0,
                "naz_speaking_time": 0,
                "total_speaking_time": 0
            }
        
        # Count Naz segments
        naz_segments = [seg for seg in segment_results if seg['is_naz']]
        total_segments = len(segment_results)
        
        # Calculate speaking time
        naz_speaking_time = sum(seg['duration'] for seg in naz_segments)
        total_speaking_time = sum(seg['duration'] for seg in segment_results)
        
        # Classification logic
        if len(naz_segments) == 0:
            classification = "lecture"
            confidence = "high"
            reason = "No Naz voice detected"
        elif len(naz_segments) >= 2 or naz_speaking_time >= 30:  # Multiple segments or significant time
            classification = "meeting"
            confidence = "high"
            reason = f"Naz detected in {len(naz_segments)} segments ({naz_speaking_time:.1f}s)"
        elif len(naz_segments) == 1 and naz_speaking_time >= 10:
            classification = "meeting"
            confidence = "medium"
            reason = f"Naz detected in 1 segment ({naz_speaking_time:.1f}s)"
        else:
            classification = "lecture"
            confidence = "medium"
            reason = f"Brief Naz detection ({naz_speaking_time:.1f}s) - likely false positive"
        
        return {
            "classification": classification,
            "confidence": confidence,
            "reason": reason,
            "naz_segments": len(naz_segments),
            "total_segments": total_segments,
            "naz_speaking_time": naz_speaking_time,
            "total_speaking_time": total_speaking_time,
            "naz_percentage": (naz_speaking_time / total_speaking_time * 100) if total_speaking_time > 0 else 0
        }
    
    def process_transcription(self, transcription_metadata):
        """Process a single transcription for Naz detection"""
        transcript_id = transcription_metadata.get('transcript_id')
        
        # Skip if already processed
        if transcript_id in self.processed_detections:
            print(f"⏭️  Skipping already processed: {transcript_id}")
            return None
        
        print(f"\n🎯 Processing: {transcription_metadata.get('topic', 'Unknown')}")
        
        # Load transcript data
        json_file = transcription_metadata.get('files', {}).get('json_file')
        if not json_file or not Path(json_file).exists():
            print(f"❌ Transcript JSON file not found: {json_file}")
            return None
        
        with open(json_file, 'r') as f:
            transcript_data = json.load(f)
        
        # Get audio file
        audio_file = transcription_metadata.get('audio_file')
        if not audio_file or not Path(audio_file).exists():
            print(f"❌ Audio file not found: {audio_file}")
            return None
        
        # Analyze segments for Naz detection
        segment_results = self.detect_naz_in_segments(transcript_data, audio_file)
        
        # Classify the recording
        classification_result = self.classify_recording(segment_results)
        
        # Mark as processed
        self.processed_detections.add(transcript_id)
        
        # Create detection metadata
        detection_metadata = {
            "transcript_id": transcript_id,
            "audio_file": audio_file,
            "user_email": transcription_metadata.get('user_email'),
            "meeting_id": transcription_metadata.get('meeting_id'),
            "topic": transcription_metadata.get('topic'),
            "start_time": transcription_metadata.get('start_time'),
            "audio_duration_seconds": transcription_metadata.get('audio_duration_seconds'),
            "classification": classification_result['classification'],
            "confidence": classification_result['confidence'],
            "reason": classification_result['reason'],
            "naz_detected": classification_result['classification'] == 'meeting',
            "naz_segments_count": classification_result['naz_segments'],
            "total_segments_count": classification_result['total_segments'],
            "naz_speaking_time_seconds": classification_result['naz_speaking_time'],
            "naz_speaking_percentage": classification_result['naz_percentage'],
            "similarity_threshold_used": self.similarity_threshold,
            "segment_details": segment_results,
            "processed_at": datetime.now().isoformat()
        }
        
        return detection_metadata
    
    def process_transcription_metadata(self, transcription_metadata_file):
        """Process all transcriptions from a metadata file"""
        transcription_metadata_file = Path(transcription_metadata_file)
        
        if not transcription_metadata_file.exists():
            print(f"❌ Transcription metadata file not found: {transcription_metadata_file}")
            return []
        
        # Ensure Naz reference embedding is ready
        if not self.create_naz_reference_embedding():
            print("❌ Cannot proceed without Naz reference embedding")
            return []
        
        # Load transcription metadata
        with open(transcription_metadata_file, 'r') as f:
            transcription_metadata_list = json.load(f)
        
        print(f"🎯 Processing {len(transcription_metadata_list)} transcriptions for Naz detection...")
        
        detection_results = []
        
        for i, transcript_meta in enumerate(transcription_metadata_list, 1):
            print(f"\n📊 Progress: {i}/{len(transcription_metadata_list)}")
            
            detection_meta = self.process_transcription(transcript_meta)
            
            if detection_meta:
                detection_results.append(detection_meta)
        
        # Save processed detections list
        self.save_processed_detections()
        
        # Save detection metadata
        detection_metadata_file = self.metadata_dir / f"naz_detection_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(detection_metadata_file, 'w') as f:
            json.dump(detection_results, f, indent=2)
        
        # Save classification summary
        self.save_classification_summary(detection_results)
        
        print(f"\n✅ Naz detection processing complete!")
        print(f"📊 Total recordings processed: {len(detection_results)}")
        print(f"📋 Detection metadata saved to: {detection_metadata_file}")
        
        return detection_results
    
    def save_classification_summary(self, detection_results):
        """Save a summary of classification results"""
        meetings = [r for r in detection_results if r['classification'] == 'meeting']
        lectures = [r for r in detection_results if r['classification'] == 'lecture']
        
        summary = {
            "total_recordings": len(detection_results),
            "meetings_with_naz": len(meetings),
            "solo_lectures": len(lectures),
            "detection_accuracy": {
                "high_confidence": len([r for r in detection_results if r['confidence'] == 'high']),
                "medium_confidence": len([r for r in detection_results if r['confidence'] == 'medium']),
                "low_confidence": len([r for r in detection_results if r['confidence'] == 'low'])
            },
            "meetings_list": [
                {
                    "topic": m['topic'],
                    "start_time": m['start_time'],
                    "naz_speaking_time": m['naz_speaking_time_seconds'],
                    "naz_percentage": m['naz_speaking_percentage']
                }
                for m in meetings
            ],
            "generated_at": datetime.now().isoformat()
        }
        
        summary_file = self.results_dir / "classification_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📈 Classification summary saved to: {summary_file}")
        print(f"🎯 Results: {len(meetings)} meetings, {len(lectures)} lectures")
    
    def process_latest_transcriptions(self):
        """Find and process the most recent transcription metadata file"""
        metadata_files = list(self.metadata_dir.glob("transcription_metadata_*.json"))
        
        if not metadata_files:
            print("❌ No transcription metadata files found. Run the transcription pipeline first.")
            return []
        
        # Get the most recent metadata file
        latest_file = max(metadata_files, key=lambda p: p.stat().st_mtime)
        print(f"📂 Using latest transcription metadata: {latest_file.name}")
        
        return self.process_transcription_metadata(latest_file)

def main():
    print("🎯 Naz Voice Detection Pipeline")
    print("===============================")
    
    # Initialize detection pipeline
    pipeline = NazDetectionPipeline()
    
    # Check for reference audio
    if not list(pipeline.reference_audio_dir.glob("*")):
        print(f"⚠️  No reference audio files found in {pipeline.reference_audio_dir}")
        print("   Please add Naz's voice samples (WAV, MP3, or M4A files) to this directory")
        print("   You can extract these from known meetings where Naz speaks")
        return
    
    # Process the latest transcriptions
    detection_results = pipeline.process_latest_transcriptions()
    
    if detection_results:
        meetings = [r for r in detection_results if r['classification'] == 'meeting']
        lectures = [r for r in detection_results if r['classification'] == 'lecture']
        
        print(f"\n🎉 Naz detection pipeline complete!")
        print(f"📊 Classification Results:")
        print(f"   🤝 Meetings with Naz: {len(meetings)}")
        print(f"   🎓 Solo lectures: {len(lectures)}")
        print(f"   📁 Results saved in: {pipeline.results_dir}")
    else:
        print("❌ No transcriptions were processed")

if __name__ == "__main__":
    main()