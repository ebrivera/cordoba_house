import streamlit as st
import json
import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import subprocess
import sys
import time

# Page config
st.set_page_config(
    page_title="Zoom Recording Pipeline Dashboard",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

class PipelineDashboard:
    def __init__(self):
        self.metadata_dir = Path("metadata")
        self.results_dir = Path("classification_results")
        self.downloads_dir = Path("downloads")
        self.audio_dir = Path("audio_files")
        self.transcripts_dir = Path("transcripts")
        
    def get_latest_metadata_file(self, pattern):
        """Get the most recent metadata file matching pattern"""
        files = list(self.metadata_dir.glob(pattern))
        if files:
            return max(files, key=lambda p: p.stat().st_mtime)
        return None
    
    def load_json_data(self, file_path):
        """Load JSON data from file"""
        if file_path and file_path.exists():
            try:
                with open(file_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                st.error(f"Error loading {file_path}: {e}")
        return None
    
    def run_pipeline_step(self, step_name, script_name):
        """Run a pipeline step with live console output"""
        try:
            # Create a container for live output
            output_container = st.empty()
            log_container = st.empty()
            
            with st.spinner(f"Running {step_name}..."):
                # Start process
                process = subprocess.Popen(
                    [sys.executable, script_name], 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                
                # Collect output in real-time
                stdout_lines = []
                stderr_lines = []
                
                # Read output line by line
                while True:
                    # Check if process has finished
                    if process.poll() is not None:
                        break
                    
                    # Read any available output
                    try:
                        import select
                        ready, _, _ = select.select([process.stdout, process.stderr], [], [], 0.1)
                        
                        if process.stdout in ready:
                            line = process.stdout.readline()
                            if line:
                                stdout_lines.append(line.strip())
                                # Show last 10 lines in real-time
                                recent_output = '\n'.join(stdout_lines[-10:])
                                output_container.code(f"🔄 Live Output:\n{recent_output}")
                        
                        if process.stderr in ready:
                            line = process.stderr.readline()
                            if line:
                                stderr_lines.append(line.strip())
                    except:
                        # Fallback for systems without select
                        time.sleep(0.5)
                
                # Get final output
                final_stdout, final_stderr = process.communicate()
                if final_stdout:
                    stdout_lines.extend(final_stdout.strip().split('\n'))
                if final_stderr:
                    stderr_lines.extend(final_stderr.strip().split('\n'))
                
                # Clear the live output
                output_container.empty()
                
                if process.returncode == 0:
                    st.success(f"✅ {step_name} completed successfully!")
                    
                    # Show summary from output
                    full_output = '\n'.join(stdout_lines)
                    
                    # Extract key metrics from output
                    metrics = self.extract_metrics_from_output(full_output, step_name)
                    if metrics:
                        col1, col2, col3 = st.columns(3)
                        for i, (key, value) in enumerate(metrics.items()):
                            with [col1, col2, col3][i % 3]:
                                st.metric(key, value)
                    
                    # Show detailed output in expander
                    with st.expander("📋 View Detailed Output"):
                        st.code(full_output, language="text")
                else:
                    st.error(f"❌ {step_name} failed! (Exit code: {process.returncode})")
                    
                    # Show error output
                    if stderr_lines:
                        st.text("❌ Error Output:")
                        st.code('\n'.join(stderr_lines), language="text")
                    
                    if stdout_lines:
                        st.text("📄 Standard Output:")
                        st.code('\n'.join(stdout_lines), language="text")
                
                return process.returncode == 0
                
        except Exception as e:
            st.error(f"❌ Error running {step_name}: {e}")
            return False
    
    def extract_metrics_from_output(self, output, step_name):
        """Extract key metrics from pipeline output"""
        metrics = {}
        
        if "download" in step_name.lower():
            # Look for download metrics
            if "downloaded" in output.lower():
                lines = output.split('\n')
                for line in lines:
                    if "downloaded" in line.lower() and "files" in line.lower():
                        # Try to extract number
                        import re
                        numbers = re.findall(r'\d+', line)
                        if numbers:
                            metrics["Files Downloaded"] = numbers[0]
                        break
        
        elif "audio" in step_name.lower():
            # Look for audio processing metrics
            if "audio files" in output.lower():
                lines = output.split('\n')
                for line in lines:
                    if "audio files created" in line.lower():
                        import re
                        numbers = re.findall(r'\d+', line)
                        if numbers:
                            metrics["Audio Files"] = numbers[0]
                        break
        
        elif "transcrib" in step_name.lower():
            # Look for transcription metrics
            if "transcribed" in output.lower():
                lines = output.split('\n')
                for line in lines:
                    if "transcribed" in line.lower() and "files" in line.lower():
                        import re
                        numbers = re.findall(r'\d+', line)
                        if numbers:
                            metrics["Transcribed"] = numbers[0]
                        break
        
        elif "detect" in step_name.lower() or "naz" in step_name.lower():
            # Look for detection metrics
            lines = output.split('\n')
            for line in lines:
                if "meetings" in line.lower() and "lectures" in line.lower():
                    import re
                    numbers = re.findall(r'\d+', line)
                    if len(numbers) >= 2:
                        metrics["Meetings"] = numbers[0]
                        metrics["Lectures"] = numbers[1]
                    break
        
        return metrics

def main():
    dashboard = PipelineDashboard()
    
    st.title("🎙️ Zoom Recording Pipeline Dashboard")
    st.markdown("Automated processing of Zoom recordings for meeting detection and transcription")
    
    # Sidebar
    st.sidebar.title("Pipeline Control")
    
    # Pipeline Steps
    st.sidebar.markdown("### Pipeline Steps")
    
    if st.sidebar.button("1️⃣ Download Recordings", type="primary"):
        dashboard.run_pipeline_step("Download Recordings", "zoom_downloader.py")
        st.rerun()
    
    if st.sidebar.button("2️⃣ Process Audio"):
        dashboard.run_pipeline_step("Audio Processing", "audio_processor.py")
        st.rerun()
    
    if st.sidebar.button("3️⃣ Transcribe Audio"):
        dashboard.run_pipeline_step("Transcription", "transcription_pipeline.py")
        st.rerun()
    
    if st.sidebar.button("4️⃣ Detect Naz Voice"):
        dashboard.run_pipeline_step("Naz Detection", "naz_detection_pipeline.py")
        st.rerun()
    
    if st.sidebar.button("🔄 Run Full Pipeline"):
        steps = [
            ("Download Recordings", "zoom_downloader.py"),
            ("Audio Processing", "audio_processor.py"),
            ("Transcription", "transcription_pipeline.py"),
            ("Naz Detection", "naz_detection_pipeline.py")
        ]
        
        for step_name, script_name in steps:
            if not dashboard.run_pipeline_step(step_name, script_name):
                st.error(f"Pipeline stopped at {step_name}")
                break
        else:
            st.success("🎉 Full pipeline completed successfully!")
        
        st.rerun()
    
    # Settings
    st.sidebar.markdown("### Settings")
    similarity_threshold = st.sidebar.slider("Naz Detection Threshold", 0.5, 0.95, 0.75, 0.05)
    
    # Main Dashboard
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "📥 Downloads", 
        "🎵 Audio Processing", 
        "📝 Transcriptions", 
        "🎯 Naz Detection"
    ])
    
    with tab1:
        st.header("📊 Pipeline Overview")
        
        # Status indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            downloads_file = dashboard.get_latest_metadata_file("downloads_*.json")
            downloads_data = dashboard.load_json_data(downloads_file) if downloads_file else []
            download_count = len(downloads_data) if downloads_data else 0
            st.metric("Downloaded Files", download_count)
        
        with col2:
            audio_file = dashboard.get_latest_metadata_file("audio_metadata_*.json")
            audio_data = dashboard.load_json_data(audio_file) if audio_file else []
            audio_count = len(audio_data) if audio_data else 0
            st.metric("Audio Files", audio_count)
        
        with col3:
            transcript_file = dashboard.get_latest_metadata_file("transcription_metadata_*.json")
            transcript_data = dashboard.load_json_data(transcript_file) if transcript_file else []
            transcript_count = len(transcript_data) if transcript_data else 0
            st.metric("Transcriptions", transcript_count)
        
        with col4:
            detection_file = dashboard.get_latest_metadata_file("naz_detection_metadata_*.json")
            detection_data = dashboard.load_json_data(detection_file) if detection_file else []
            detection_count = len(detection_data) if detection_data else 0
            st.metric("Analyzed Recordings", detection_count)
        
        # Summary chart
        if detection_data:
            meetings = [r for r in detection_data if r['classification'] == 'meeting']
            lectures = [r for r in detection_data if r['classification'] == 'lecture']
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Classification pie chart
                fig = px.pie(
                    values=[len(meetings), len(lectures)],
                    names=['Meetings with Naz', 'Solo Lectures'],
                    title="Recording Classification"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Confidence distribution
                confidence_counts = {}
                for r in detection_data:
                    conf = r.get('confidence', 'unknown')
                    confidence_counts[conf] = confidence_counts.get(conf, 0) + 1
                
                fig = px.bar(
                    x=list(confidence_counts.keys()),
                    y=list(confidence_counts.values()),
                    title="Detection Confidence Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("📥 Downloaded Recordings")
        
        downloads_file = dashboard.get_latest_metadata_file("downloads_*.json")
        downloads_data = dashboard.load_json_data(downloads_file)
        
        if downloads_data:
            df = pd.DataFrame(downloads_data)
            
            # Summary stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Files", len(df))
            with col2:
                total_size_gb = df['file_size'].sum() / (1024**3)
                st.metric("Total Size", f"{total_size_gb:.2f} GB")
            with col3:
                unique_meetings = df['meeting_id'].nunique()
                st.metric("Unique Meetings", unique_meetings)
            
            # File type distribution
            type_counts = df['recording_type'].value_counts()
            fig = px.bar(x=type_counts.index, y=type_counts.values, title="Recording Type Distribution")
            st.plotly_chart(fig, use_container_width=True)
            
            # Downloads table
            st.subheader("Downloaded Files")
            display_df = df[['topic', 'start_time', 'recording_type', 'file_type', 'file_size', 'user_email']].copy()
            display_df['file_size_mb'] = (display_df['file_size'] / (1024**2)).round(2)
            display_df['date'] = pd.to_datetime(display_df['start_time']).dt.date
            
            st.dataframe(
                display_df[['date', 'topic', 'recording_type', 'file_type', 'file_size_mb', 'user_email']],
                use_container_width=True
            )
        else:
            st.info("No download data available. Run the download step first.")
    
    with tab3:
        st.header("🎵 Audio Processing")
        
        audio_file = dashboard.get_latest_metadata_file("audio_metadata_*.json")
        audio_data = dashboard.load_json_data(audio_file)
        
        if audio_data:
            df = pd.DataFrame(audio_data)
            
            # Summary stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Audio Files", len(df))
            with col2:
                total_duration = df['audio_duration_seconds'].sum() / 3600  # Convert to hours
                st.metric("Total Duration", f"{total_duration:.1f} hours")
            with col3:
                avg_duration = df['audio_duration_seconds'].mean() / 60  # Convert to minutes
                st.metric("Avg Duration", f"{avg_duration:.1f} minutes")
            
            # Duration distribution
            fig = px.histogram(
                df, 
                x=df['audio_duration_seconds'] / 60,  # Convert to minutes
                title="Audio Duration Distribution (minutes)",
                nbins=20
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Audio files table
            st.subheader("Processed Audio Files")
            display_df = df[['topic', 'start_time', 'recording_type', 'audio_duration_seconds', 'user_email']].copy()
            display_df['duration_minutes'] = (display_df['audio_duration_seconds'] / 60).round(1)
            display_df['date'] = pd.to_datetime(display_df['start_time']).dt.date
            
            st.dataframe(
                display_df[['date', 'topic', 'recording_type', 'duration_minutes', 'user_email']],
                use_container_width=True
            )
        else:
            st.info("No audio processing data available. Run the audio processing step first.")
    
    with tab4:
        st.header("📝 Transcriptions")
        
        transcript_file = dashboard.get_latest_metadata_file("transcription_metadata_*.json")
        transcript_data = dashboard.load_json_data(transcript_file)
        
        if transcript_data:
            df = pd.DataFrame(transcript_data)
            
            # Summary stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Transcribed Files", len(df))
            with col2:
                avg_confidence = df['confidence'].mean()
                st.metric("Avg Confidence", f"{avg_confidence:.2f}")
            with col3:
                multi_speaker = df[df['speaker_count'] > 1]
                st.metric("Multi-Speaker", len(multi_speaker))
            
            # Speaker count distribution
            fig = px.histogram(df, x='speaker_count', title="Speaker Count Distribution")
            st.plotly_chart(fig, use_container_width=True)
            
            # Transcription table
            st.subheader("Transcription Results")
            display_df = df[['topic', 'start_time', 'speaker_count', 'confidence', 'transcript_length', 'user_email']].copy()
            display_df['date'] = pd.to_datetime(display_df['start_time']).dt.date
            
            st.dataframe(
                display_df[['date', 'topic', 'speaker_count', 'confidence', 'transcript_length', 'user_email']],
                use_container_width=True
            )
            
            # View specific transcripts
            st.subheader("View Transcript")
            selected_topic = st.selectbox("Select a recording to view transcript:", df['topic'].tolist())
            
            if selected_topic:
                selected_row = df[df['topic'] == selected_topic].iloc[0]
                transcript_file_path = selected_row['files']['transcript_file']
                
                if Path(transcript_file_path).exists():
                    with open(transcript_file_path, 'r', encoding='utf-8') as f:
                        transcript_text = f.read()
                    
                    st.text_area("Transcript", transcript_text, height=300)
                    
                    # Show detailed transcript if available
                    detailed_file = selected_row['files'].get('detailed_file')
                    if detailed_file and Path(detailed_file).exists():
                        with st.expander("View Detailed Transcript with Speaker Labels"):
                            with open(detailed_file, 'r', encoding='utf-8') as f:
                                detailed_text = f.read()
                            st.text_area("Detailed Transcript", detailed_text, height=400)
        else:
            st.info("No transcription data available. Run the transcription step first.")
    
    with tab5:
        st.header("🎯 Naz Detection Results")
        
        detection_file = dashboard.get_latest_metadata_file("naz_detection_metadata_*.json")
        detection_data = dashboard.load_json_data(detection_file)
        
        if detection_data:
            df = pd.DataFrame(detection_data)
            
            # Classification summary
            meetings = df[df['classification'] == 'meeting']
            lectures = df[df['classification'] == 'lecture']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Recordings", len(df))
            with col2:
                st.metric("Meetings with Naz", len(meetings))
            with col3:
                st.metric("Solo Lectures", len(lectures))
            with col4:
                high_conf = df[df['confidence'] == 'high']
                st.metric("High Confidence", len(high_conf))
            
            # Naz speaking time distribution for meetings
            if len(meetings) > 0:
                fig = px.bar(
                    meetings,
                    x='topic',
                    y='naz_speaking_percentage',
                    title="Naz Speaking Time % in Meetings",
                    labels={'naz_speaking_percentage': 'Naz Speaking %'}
                )
                fig.update_xaxis(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            
            # Results table
            st.subheader("Detection Results")
            display_df = df[['topic', 'start_time', 'classification', 'confidence', 'naz_speaking_time_seconds', 'naz_speaking_percentage', 'user_email']].copy()
            display_df['date'] = pd.to_datetime(display_df['start_time']).dt.date
            display_df['naz_time_min'] = (display_df['naz_speaking_time_seconds'] / 60).round(1)
            display_df['naz_pct'] = display_df['naz_speaking_percentage'].round(1)
            
            # Color code by classification
            def highlight_classification(row):
                if row['classification'] == 'meeting':
                    return ['background-color: #d4edda'] * len(row)
                else:
                    return ['background-color: #f8d7da'] * len(row)
            
            styled_df = display_df[['date', 'topic', 'classification', 'confidence', 'naz_time_min', 'naz_pct', 'user_email']].style.apply(highlight_classification, axis=1)
            st.dataframe(styled_df, use_container_width=True)
            
            # Meetings summary
            if len(meetings) > 0:
                st.subheader("📅 Identified Meetings with Naz")
                for _, meeting in meetings.iterrows():
                    with st.expander(f"{meeting['topic']} - {meeting['start_time'][:10]}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Duration:** {meeting['audio_duration_seconds']/60:.1f} minutes")
                            st.write(f"**Naz Speaking Time:** {meeting['naz_speaking_time_seconds']/60:.1f} minutes")
                            st.write(f"**Naz Speaking %:** {meeting['naz_speaking_percentage']:.1f}%")
                        
                        with col2:
                            st.write(f"**Confidence:** {meeting['confidence']}")
                            st.write(f"**Naz Segments:** {meeting['naz_segments_count']}")
                            st.write(f"**Total Segments:** {meeting['total_segments_count']}")
                        
                        st.write(f"**Reason:** {meeting['reason']}")
        else:
            st.info("No detection data available. Run the Naz detection step first.")
        
        # Summary file
        summary_file = dashboard.results_dir / "classification_summary.json"
        if summary_file.exists():
            summary_data = dashboard.load_json_data(summary_file)
            if summary_data:
                st.subheader("📈 Classification Summary")
                st.json(summary_data)

if __name__ == "__main__":
    main()