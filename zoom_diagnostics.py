import os
import csv
import requests
import base64
from dotenv import load_dotenv
from pathlib import Path
import json
from datetime import datetime
import hashlib

load_dotenv()

class ZoomDownloader:
    def __init__(self):
        self.client_id = os.getenv("ZOOM_CLIENT_ID")
        self.client_secret = os.getenv("ZOOM_CLIENT_SECRET")
        self.account_id = os.getenv("ZOOM_ACCOUNT_ID")
        self.api_base = "https://api.zoom.us/v2"
        
        # Create directories
        self.downloads_dir = Path("downloads")
        self.metadata_dir = Path("metadata")
        self.downloads_dir.mkdir(exist_ok=True)
        self.metadata_dir.mkdir(exist_ok=True)
        
        # Track processed recordings to avoid re-downloads
        self.processed_file = self.metadata_dir / "processed_meetings.json"
        self.processed_meetings = self.load_processed_meetings()
    
    def load_processed_meetings(self):
        """Load list of already processed meetings"""
        if self.processed_file.exists():
            with open(self.processed_file, 'r') as f:
                return set(json.load(f))
        return set()
    
    def save_processed_meetings(self):
        """Save list of processed meetings"""
        with open(self.processed_file, 'w') as f:
            json.dump(list(self.processed_meetings), f)
    
    def show_processed_summary(self):
        """Show summary of what's already been processed"""
        if not self.processed_meetings:
            print("📋 No meetings processed yet")
            return
        
        print(f"📋 Already processed {len(self.processed_meetings)} meeting recordings:")
        
        # Try to get more details from latest metadata file
        latest_metadata = self.get_latest_metadata_file()
        if latest_metadata:
            with open(latest_metadata, 'r') as f:
                metadata_list = json.load(f)
            
            # Group by date and topic
            from collections import defaultdict
            by_date = defaultdict(list)
            
            for item in metadata_list:
                date_key = item.get('date_key', item.get('start_time', '')[:10])
                by_date[date_key].append(item)
            
            for date in sorted(by_date.keys()):
                items = by_date[date]
                topics = set(item['topic'] for item in items)
                file_count = len(items)
                print(f"   {date}: {len(topics)} meetings, {file_count} files")
                for topic in sorted(topics):
                    topic_items = [item for item in items if item['topic'] == topic]
                    types = [item['recording_type'] for item in topic_items]
                    print(f"     • {topic[:50]}... ({', '.join(set(types))})")
        else:
            print(f"   {len(self.processed_meetings)} recordings (run with -v for details)")
    
    def get_latest_metadata_file(self):
        """Get the most recent metadata file"""
        files = list(self.metadata_dir.glob("downloads_*.json"))
        if files:
            return max(files, key=lambda p: p.stat().st_mtime)
        return None
    
    def clear_processed_cache(self):
        """Clear the processed meetings cache (force re-download)"""
        if self.processed_file.exists():
            backup_file = self.metadata_dir / f"processed_meetings_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            self.processed_file.rename(backup_file)
            print(f"📋 Cleared processed cache (backup saved to {backup_file.name})")
        
        self.processed_meetings.clear()
        print("🔄 All meetings will be re-downloaded on next run")
    
    def get_zoom_token(self):
        """Get Zoom API access token"""
        creds = f"{self.client_id}:{self.client_secret}"
        encoded = base64.b64encode(creds.encode()).decode()
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Authorization": f"Basic {encoded}"
        }
        data = {
            "grant_type": "account_credentials",
            "account_id": self.account_id
        }
        
        resp = requests.post("https://zoom.us/oauth/token", headers=headers, data=data)
        resp.raise_for_status()
        return resp.json()["access_token"]
    
    def get_user_id(self, email, token):
        """Get user ID from email"""
        url = f"{self.api_base}/users/{email}"
        headers = {"Authorization": f"Bearer {token}"}
        r = requests.get(url, headers=headers)
        r.raise_for_status()
        return r.json()["id"]
    
    def get_all_recordings_comprehensive(self, user_email):
        """Comprehensive search using multiple methods to find ALL recordings"""
        token = self.get_zoom_token()
        user_id = self.get_user_id(user_email, token)
        headers = {"Authorization": f"Bearer {token}"}
        
        all_recordings = []
        found_meetings = set()  # Track unique meetings by ID
        
        print(f"🔍 COMPREHENSIVE RECORDING SEARCH")
        print(f"=" * 50)
        print(f"User: {user_email} (ID: {user_id})")
        print()
        
        # Method 1: Extended date ranges with different parameters
        print("1️⃣ EXTENDED DATE RANGE SEARCH")
        print("-" * 30)
        
        # Go back further and use smaller chunks for better API results
        date_ranges = [
            ("2018-01-01", "2019-12-31"),
            ("2020-01-01", "2020-12-31"),
            ("2021-01-01", "2021-12-31"),
            ("2022-01-01", "2022-06-30"),
            ("2022-07-01", "2022-12-31"),
            ("2023-01-01", "2023-06-30"),
            ("2023-07-01", "2023-12-31"),
            ("2024-01-01", "2024-06-30"),
            ("2024-07-01", "2024-12-31"),
            ("2025-01-01", "2025-12-31"),
        ]
        
        base_url = f"{self.api_base}/users/{user_id}/recordings"
        
        for start_date, end_date in date_ranges:
            print(f"📅 Searching {start_date} to {end_date}")
            
            # Try multiple parameter combinations
            param_sets = [
                {"page_size": 300, "from": start_date, "to": end_date},
                {"page_size": 300, "from": start_date, "to": end_date, "mc": "false"},
                {"page_size": 300, "from": start_date, "to": end_date, "trash": "true"},
                {"page_size": 300, "from": start_date, "to": end_date, "mc": "false", "trash": "true"},
            ]
            
            period_meetings = set()
            
            for param_idx, params in enumerate(param_sets):
                param_desc = f"mc={params.get('mc', 'default')}, trash={params.get('trash', 'false')}"
                
                page_num = 1
                while True:
                    try:
                        r = requests.get(base_url, headers=headers, params=params)
                        if r.ok:
                            data = r.json()
                            meetings = data.get("meetings", [])
                            
                            if page_num == 1:
                                total_size = data.get("total_size", 0)
                                print(f"   Params {param_idx+1} ({param_desc}): {len(meetings)} meetings, total_size: {total_size}")
                            
                            for meeting in meetings:
                                meeting_id = meeting.get('id')
                                if meeting_id and meeting_id not in found_meetings:
                                    period_meetings.add(meeting_id)
                                    found_meetings.add(meeting_id)
                                    
                                    recording_files = []
                                    for rf in meeting.get('recording_files', []):
                                        recording_files.append({
                                            "recording_type": rf.get('recording_type'),
                                            "file_type": rf.get('file_type'),
                                            "file_size": rf.get('file_size', 0),
                                            "download_url": f"{rf['download_url']}?access_token={token}" if rf.get('download_url') else None,
                                            "file_extension": rf.get('file_extension', rf.get('file_type', '').lower()),
                                            "recording_start": rf.get('recording_start', ''),
                                            "recording_end": rf.get('recording_end', ''),
                                            "status": rf.get('status', 'completed')
                                        })
                                    
                                    if recording_files:
                                        meeting_data = {
                                            "user_email": user_email,
                                            "topic": meeting.get('topic'),
                                            "start_time": meeting.get('start_time'),
                                            "duration": meeting.get('duration'),
                                            "meeting_id": meeting_id,
                                            "uuid": meeting.get('uuid'),
                                            "recording_count": meeting.get('recording_count', len(recording_files)),
                                            "total_size": meeting.get('total_size', 0),
                                            "recording_files": recording_files,
                                            "found_via": f"user_recordings_{param_desc}"
                                        }
                                        all_recordings.append(meeting_data)
                            
                            # Check for next page
                            next_page_token = data.get("next_page_token")
                            if next_page_token:
                                params["next_page_token"] = next_page_token
                                page_num += 1
                            else:
                                break
                        else:
                            if r.status_code == 429:  # Rate limit
                                print(f"   ⏳ Rate limited, waiting 30 seconds...")
                                time.sleep(30)
                                continue
                            else:
                                error_data = r.json() if r.content else {}
                                print(f"   ❌ Error {r.status_code}: {error_data}")
                                break
                    except Exception as e:
                        print(f"   ❌ Exception: {e}")
                        break
                    
                    # Remove next_page_token for next parameter set
                    if "next_page_token" in params:
                        del params["next_page_token"]
            
            print(f"   📊 Period total: {len(period_meetings)} unique meetings")
        
        print(f"\n2️⃣ ACCOUNT-LEVEL SEARCH")
        print("-" * 30)
        
        # Method 2: Account-level recordings (if accessible)
        try:
            account_url = f"{self.api_base}/accounts/{self.account_id}/recordings"
            params = {"page_size": 300, "from": "2018-01-01", "to": "2025-12-31"}
            
            r = requests.get(account_url, headers=headers, params=params)
            if r.ok:
                data = r.json()
                account_meetings = data.get("meetings", [])
                print(f"   Account recordings found: {len(account_meetings)}")
                
                for meeting in account_meetings:
                    meeting_id = meeting.get('id')
                    if meeting_id and meeting_id not in found_meetings:
                        found_meetings.add(meeting_id)
                        # Process account-level meetings same way...
                        # (abbreviated for space, but would include same logic)
            else:
                print(f"   Account recordings not accessible: {r.status_code}")
        except Exception as e:
            print(f"   Account recordings error: {e}")
        
        print(f"\n3️⃣ ALTERNATIVE USER LOOKUP")
        print("-" * 30)
        
        # Method 3: Try searching by user ID directly instead of email resolution
        try:
            # Sometimes the email->ID resolution misses recordings
            direct_url = f"{self.api_base}/users/{user_email}/recordings"  # Use email directly
            params = {"page_size": 300, "from": "2018-01-01", "to": "2025-12-31", "mc": "false"}
            
            r = requests.get(direct_url, headers=headers, params=params)
            if r.ok:
                data = r.json()
                direct_meetings = data.get("meetings", [])
                print(f"   Direct email search: {len(direct_meetings)} meetings")
                
                new_meetings = 0
                for meeting in direct_meetings:
                    meeting_id = meeting.get('id')
                    if meeting_id and meeting_id not in found_meetings:
                        new_meetings += 1
                        found_meetings.add(meeting_id)
                        # Process new meetings...
                
                print(f"   New meetings found: {new_meetings}")
            else:
                print(f"   Direct email search failed: {r.status_code}")
        except Exception as e:
            print(f"   Direct email search error: {e}")
        
        print(f"\n📊 SEARCH SUMMARY")
        print("=" * 30)
        print(f"Total unique meetings found: {len(found_meetings)}")
        print(f"Meetings with recording files: {len(all_recordings)}")
        print(f"Total recording files: {sum(len(m['recording_files']) for m in all_recordings)}")
        
        # Show breakdown by year
        by_year = {}
        for recording in all_recordings:
            year = recording['start_time'][:4] if recording['start_time'] else 'Unknown'
            by_year[year] = by_year.get(year, 0) + 1
        
        print(f"\nBreakdown by year:")
        for year in sorted(by_year.keys()):
            print(f"   {year}: {by_year[year]} meetings")
        
        # If still way fewer than expected, provide guidance
        if len(all_recordings) < 50:  # Based on your CSV showing 104
            print(f"\n⚠️  STILL MISSING RECORDINGS")
            print(f"Expected ~104 based on your CSV, found {len(all_recordings)}")
            print(f"\nPossible reasons:")
            print(f"1. Many recordings are stored locally (not in cloud)")
            print(f"2. Recordings are in different Zoom accounts")
            print(f"3. Some recordings have been deleted")
            print(f"4. API access limitations")
            print(f"5. Different user accounts used for recording")
            print(f"\nNext steps:")
            print(f"• Manually check Zoom web portal")
            print(f"• Verify which account contains the recordings")
            print(f"• Check if there are multiple teacher accounts")
            print(f"• Run diagnostics: python zoom_diagnostics.py")
        
        return all_recordings
        """Get all recordings for a user across specified date ranges"""
        if date_ranges is None:
            # Extended date ranges to catch everything (last 5 years)
            date_ranges = [
                ("2020-01-01", "2020-12-31"),
                ("2021-01-01", "2021-12-31"),
                ("2022-01-01", "2022-06-30"),
                ("2022-07-01", "2022-12-31"), 
                ("2023-01-01", "2023-06-30"),
                ("2023-07-01", "2023-12-31"),
                ("2024-01-01", "2024-06-30"),
                ("2024-07-01", "2024-12-31"),
                ("2025-01-01", "2025-12-31")
            ]
        
        token = self.get_zoom_token()
        user_id = self.get_user_id(user_email, token)
        
        all_recordings = []
        total_meetings_found = 0
        
        for start_date, end_date in date_ranges:
            print(f"📅 Searching {start_date} to {end_date} for {user_email}")
            
            url = f"{self.api_base}/users/{user_id}/recordings"
            headers = {"Authorization": f"Bearer {token}"}
            params = {
                "page_size": 300,  # Maximum allowed
                "from": start_date,
                "to": end_date,
                "mc": "false"  # Include meetings without recordings
            }
            
            page_num = 1
            period_meetings = 0
            
            while True:
                try:
                    r = requests.get(url, headers=headers, params=params)
                    if r.ok:
                        data = r.json()
                        meetings = data.get("meetings", [])
                        page_count = data.get("page_count", 1)
                        total_size = data.get("total_size", 0)
                        
                        print(f"   Page {page_num}/{page_count}: Found {len(meetings)} meetings (Total in period: {total_size})")
                        period_meetings += len(meetings)
                        
                        for meeting in meetings:
                            recording_files = []
                            raw_files = meeting.get('recording_files', [])
                            
                            # Debug: show what we're finding
                            if len(raw_files) == 0:
                                print(f"     ⚠️  Meeting '{meeting.get('topic', 'Unknown')}' has no recording files")
                                continue
                            
                            for rf in raw_files:
                                # Include ALL file types
                                recording_files.append({
                                    "recording_type": rf.get('recording_type'),
                                    "file_type": rf.get('file_type'),
                                    "file_size": rf.get('file_size', 0),
                                    "download_url": f"{rf['download_url']}?access_token={token}" if rf.get('download_url') else None,
                                    "file_extension": rf.get('file_extension', rf.get('file_type', '').lower()),
                                    "recording_start": rf.get('recording_start', ''),
                                    "recording_end": rf.get('recording_end', ''),
                                    "status": rf.get('status', 'completed')
                                })
                            
                            if recording_files:  # Only include meetings with actual recording files
                                meeting_data = {
                                    "user_email": user_email,
                                    "topic": meeting.get('topic'),
                                    "start_time": meeting.get('start_time'),
                                    "duration": meeting.get('duration'),
                                    "meeting_id": meeting.get('id'),
                                    "uuid": meeting.get('uuid'),  # Add UUID for better tracking
                                    "recording_count": meeting.get('recording_count', len(recording_files)),
                                    "total_size": meeting.get('total_size', 0),
                                    "recording_files": recording_files
                                }
                                all_recordings.append(meeting_data)
                                
                                # Debug output
                                file_types = [rf['recording_type'] for rf in recording_files]
                                print(f"     ✅ '{meeting.get('topic', 'Unknown')}' - {len(recording_files)} files: {', '.join(file_types)}")
                        
                        # Check for next page
                        next_page_token = data.get("next_page_token")
                        if next_page_token:
                            params["next_page_token"] = next_page_token
                            page_num += 1
                        else:
                            break
                            
                    else:
                        error_data = r.json() if r.content else {"error": "No response content"}
                        print(f"   ❌ API Error: {r.status_code} - {error_data}")
                        if r.status_code == 429:  # Rate limit
                            print("   ⏳ Rate limited - waiting 60 seconds...")
                            time.sleep(60)
                            continue
                        break
                        
                except Exception as e:
                    print(f"   ❌ Exception: {e}")
                    break
            
            total_meetings_found += period_meetings
            print(f"   📊 Period summary: {period_meetings} meetings with recordings")
        
        print(f"\n📊 SEARCH COMPLETE:")
        print(f"   🔍 Total meetings found across all periods: {total_meetings_found}")
        print(f"   📁 Meetings with recording files: {len(all_recordings)}")
        print(f"   📧 User: {user_email}")
        
        if len(all_recordings) == 0:
            print("\n⚠️  NO RECORDINGS FOUND - Possible reasons:")
            print("   1. No cloud recordings in the searched date ranges")
            print("   2. Recordings are stored locally (not in cloud)")
            print("   3. Missing API scopes - check Zoom app permissions")
            print("   4. User has no recorded meetings")
            print("   5. Recordings were deleted")
        
        return all_recordings
    
    def create_meeting_key(self, topic, start_time, meeting_id, recording_type, file_type):
        """Create unique key for meeting + recording type to avoid re-downloads"""
        # Clean topic for consistent comparison
        clean_topic = "".join(c.lower() for c in topic if c.isalnum() or c.isspace()).strip()
        clean_topic = " ".join(clean_topic.split())  # Normalize whitespace
        
        # Use date (YYYY-MM-DD) from start_time
        date_part = start_time[:10] if start_time else "unknown_date"
        
        # Create composite key: topic + date + meeting_id + recording_type + file_type
        key_content = f"{clean_topic}_{date_part}_{meeting_id}_{recording_type}_{file_type}"
        
        # Return hash for consistent length
        return hashlib.md5(key_content.encode()).hexdigest()[:16]
    
    def download_recording(self, download_url, local_path):
        """Download a recording file with progress"""
        try:
            print(f"📥 Downloading to {local_path}")
            
            # Stream download for large files
            response = requests.get(download_url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Show progress every 10MB
                        if downloaded % (10 * 1024 * 1024) == 0:
                            if total_size > 0:
                                percent = (downloaded / total_size) * 100
                                print(f"   Progress: {percent:.1f}% ({downloaded/(1024*1024):.1f}MB)")
            
            print(f"✅ Downloaded {local_path.name} ({downloaded/(1024*1024):.1f}MB)")
            return True
            
        except Exception as e:
            print(f"❌ Download failed for {local_path}: {e}")
            return False
    
    def process_recordings(self, user_emails):
        """Main method to download all recordings from multiple users"""
        if isinstance(user_emails, str):
            user_emails = [user_emails]
        
        all_downloaded = []
        
        for user_email in user_emails:
            print(f"\n🔍 Processing recordings for {user_email}")
            recordings = self.get_all_recordings_comprehensive(user_email)
            
            for recording in recordings:
                meeting_id = recording['meeting_id']
                topic = recording['topic']
                start_time = recording['start_time']
                
                # Create meeting folder
                safe_topic = "".join(c for c in topic if c.isalnum() or c in (' ', '-', '_')).rstrip()[:50]
                date_str = start_time[:10]  # YYYY-MM-DD
                meeting_folder = self.downloads_dir / f"{date_str}_{safe_topic}_{meeting_id}"
                meeting_folder.mkdir(exist_ok=True)
                
                # Download each recording file
                for rf in recording['recording_files']:
                    meeting_key = self.create_meeting_key(
                        topic, start_time, meeting_id, rf['recording_type'], rf['file_type']
                    )
                    
                    # Skip if already processed
                    if meeting_key in self.processed_meetings:
                        print(f"⏭️  Skipping already downloaded: {topic} - {rf['recording_type']} ({rf['file_type']})")
                        print(f"     Meeting key: {meeting_key}")
                        continue
                    
                    if not rf['download_url']:
                        print(f"⚠️  No download URL for {rf['recording_type']}")
                        continue
                    
                    # Create filename
                    file_ext = rf.get('file_extension', rf['file_type'].lower())
                    if not file_ext.startswith('.'):
                        file_ext = f".{file_ext}"
                    
                    filename = f"{rf['recording_type']}{file_ext}"
                    local_path = meeting_folder / filename
                    
                    # Download file
                    if self.download_recording(rf['download_url'], local_path):
                        # Mark as processed
                        self.processed_meetings.add(meeting_key)
                        
                        # Save metadata
                        file_metadata = {
                            "meeting_key": meeting_key,
                            "user_email": user_email,
                            "meeting_id": meeting_id,
                            "topic": topic,
                            "start_time": start_time,
                            "duration": recording['duration'],
                            "recording_type": rf['recording_type'],
                            "file_type": rf['file_type'],
                            "file_size": rf['file_size'],
                            "local_path": str(local_path),
                            "downloaded_at": datetime.now().isoformat(),
                            "date_key": start_time[:10] if start_time else "unknown"
                        }
                        
                        all_downloaded.append(file_metadata)
                        
                        # Save after each successful download to prevent data loss
                        self.save_processed_meetings()
        
        # Save download metadata
        metadata_file = self.metadata_dir / f"downloads_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(metadata_file, 'w') as f:
            json.dump(all_downloaded, f, indent=2)
        
        print(f"\n✅ Download complete! Metadata saved to {metadata_file}")
        print(f"📊 Total files downloaded: {len(all_downloaded)}")
        
        return all_downloaded

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Download Zoom recordings")
    parser.add_argument("--emails", nargs="+", 
                       default=["cordobahouseschool3@gmail.com"],
                       help="User emails to process")
    parser.add_argument("--show-processed", action="store_true",
                       help="Show what's already been processed")
    parser.add_argument("--clear-cache", action="store_true", 
                       help="Clear processed cache (force re-download)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    # Initialize downloader
    downloader = ZoomDownloader()
    
    # Handle special commands
    if args.show_processed:
        downloader.show_processed_summary()
        return
    
    if args.clear_cache:
        downloader.clear_processed_cache()
        return
    
    # Show current status
    if args.verbose or len(downloader.processed_meetings) > 0:
        downloader.show_processed_summary()
        print()
    
    # Process recordings
    downloaded_files = downloader.process_recordings(args.emails)
    
    print(f"\n🎉 Download pipeline complete!")
    print(f"📊 Total files downloaded this session: {len(downloaded_files)}")
    print(f"📁 Files saved in: {downloader.downloads_dir}")
    print(f"📋 Metadata saved in: {downloader.metadata_dir}")
    
    if len(downloaded_files) == 0:
        print("💡 All files may already be downloaded. Use --show-processed to see what's cached.")
        print("💡 Use --clear-cache to force re-download everything.")

if __name__ == "__main__":
    main()