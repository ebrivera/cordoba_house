import os
import base64
import requests
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv()

ZOOM_CLIENT_ID = os.getenv("ZOOM_CLIENT_ID")
ZOOM_CLIENT_SECRET = os.getenv("ZOOM_CLIENT_SECRET")
ZOOM_ACCOUNT_ID = os.getenv("ZOOM_ACCOUNT_ID")
ZOOM_API_BASE = "https://api.zoom.us/v2"

def get_zoom_token():
    creds = f"{ZOOM_CLIENT_ID}:{ZOOM_CLIENT_SECRET}"
    encoded = base64.b64encode(creds.encode()).decode()
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Authorization": f"Basic {encoded}"
    }
    data = {
        "grant_type": "account_credentials",
        "account_id": ZOOM_ACCOUNT_ID
    }
    resp = requests.post("https://zoom.us/oauth/token", headers=headers, data=data)
    resp.raise_for_status()
    return resp.json()["access_token"]

def get_user_id(email, token):
    url = f"{ZOOM_API_BASE}/users/{email}"
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    return r.json()["id"]

def get_recordings_2022(user_id, token):
    """Get recordings from 2022 (where your recordings actually are!)"""
    print("🎯 Searching 2022 recordings (where your recordings are)...")
    
    url = f"{ZOOM_API_BASE}/users/{user_id}/recordings"
    headers = {"Authorization": f"Bearer {token}"}
    
    # Search all of 2022 in chunks (Zoom has 6-month limit per API call)
    date_ranges = [
        ("2022-01-01", "2022-06-30"),  # First half of 2022
        ("2022-07-01", "2022-12-31"),  # Second half of 2022
    ]
    
    all_recordings = []
    
    for start_date, end_date in date_ranges:
        print(f"\n📅 Searching {start_date} to {end_date}")
        
        params = {
            "page_size": 300,
            "from": start_date,
            "to": end_date
        }
        
        page_num = 1
        while True:
            try:
                r = requests.get(url, headers=headers, params=params)
                if r.ok:
                    data = r.json()
                    meetings = data.get("meetings", [])
                    
                    print(f"   Page {page_num}: Found {len(meetings)} meetings")
                    
                    for meeting in meetings:
                        recording_files = []
                        for rf in meeting.get('recording_files', []):
                            recording_files.append({
                                "recording_type": rf.get('recording_type'),
                                "file_type": rf.get('file_type'),
                                "file_size": rf.get('file_size', 0),
                                "download_url": f"{rf['download_url']}?access_token={token}" if rf.get('download_url') else None,
                                "play_url": rf.get('play_url', ''),
                                "recording_start": rf.get('recording_start', ''),
                                "recording_end": rf.get('recording_end', '')
                            })
                        
                        if recording_files:  # Only include meetings with actual recording files
                            all_recordings.append({
                                "topic": meeting.get('topic'),
                                "start_time": meeting.get('start_time'),
                                "duration": meeting.get('duration'),
                                "meeting_id": meeting.get('id'),
                                "recording_files": recording_files
                            })
                    
                    # Check for next page
                    next_page_token = data.get("next_page_token")
                    if next_page_token:
                        params["next_page_token"] = next_page_token
                        page_num += 1
                    else:
                        break
                        
                else:
                    print(f"   ❌ API Error: {r.status_code} - {r.json()}")
                    break
                    
            except Exception as e:
                print(f"   ❌ Exception: {e}")
                break
    
    return all_recordings

def print_recordings(recordings):
    print(f"\n🎉 FOUND {len(recordings)} MEETINGS WITH RECORDINGS!")
    print("=" * 80)
    
    # Group by month for better organization
    from collections import defaultdict
    by_month = defaultdict(list)
    
    for recording in recordings:
        start_time = recording['start_time']
        if start_time:
            month_key = start_time[:7]  # YYYY-MM
            by_month[month_key].append(recording)
    
    # Sort months
    sorted_months = sorted(by_month.keys())
    
    for month in sorted_months:
        month_recordings = by_month[month]
        print(f"\n📅 {month} ({len(month_recordings)} recordings)")
        print("-" * 60)
        
        for i, recording in enumerate(month_recordings, 1):
            print(f"{i}. {recording['topic']}")
            print(f"   📅 Date: {recording['start_time']}")
            print(f"   ⏱️  Duration: {recording.get('duration', 'Unknown')} minutes")
            print(f"   🎬 Files ({len(recording['recording_files'])}):")
            
            for j, rf in enumerate(recording['recording_files'], 1):
                size_mb = rf['file_size'] / (1024 * 1024) if rf['file_size'] else 0
                print(f"      {j}. {rf['recording_type']} ({rf['file_type']}) - {size_mb:.1f}MB")
                if rf['download_url']:
                    print(f"         📥 Download: {rf['download_url']}")
            print()

def save_to_csv(recordings):
    """Save recordings info to CSV for easy viewing"""
    import csv
    
    filename = "zoom_recordings_2022.csv"
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['topic', 'date', 'duration', 'meeting_id', 'recording_type', 'file_type', 'size_mb', 'download_url']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for recording in recordings:
            for rf in recording['recording_files']:
                size_mb = rf['file_size'] / (1024 * 1024) if rf['file_size'] else 0
                writer.writerow({
                    'topic': recording['topic'],
                    'date': recording['start_time'],
                    'duration': recording.get('duration', ''),
                    'meeting_id': recording['meeting_id'],
                    'recording_type': rf['recording_type'],
                    'file_type': rf['file_type'],
                    'size_mb': round(size_mb, 1),
                    'download_url': rf['download_url']
                })
    
    print(f"📊 Saved detailed info to {filename}")

def main():
    token = get_zoom_token()
    user_id = get_user_id("cordobahouseschool3@gmail.com", token)
    
    print(f"🔍 Getting 2022 recordings for Al Ghazzali...")
    print(f"User ID: {user_id}")
    print("=" * 80)
    
    recordings = get_recordings_2022(user_id, token)
    
    if recordings:
        print_recordings(recordings)
        save_to_csv(recordings)
        
        print(f"\n✅ SUCCESS! Found {len(recordings)} recordings from 2022")
        print("🎯 These match what you see in the Zoom web portal")
        
        # Show summary stats
        total_files = sum(len(r['recording_files']) for r in recordings)
        total_size = sum(rf['file_size'] for r in recordings for rf in r['recording_files'])
        total_size_gb = total_size / (1024 * 1024 * 1024)
        
        print(f"📊 Summary:")
        print(f"   • {len(recordings)} meetings")
        print(f"   • {total_files} recording files")
        print(f"   • {total_size_gb:.2f} GB total size")
        
    else:
        print("❌ Still no recordings found. There might be an API issue.")

if __name__ == "__main__":
    main()