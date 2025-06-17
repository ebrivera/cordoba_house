import os
import json
import time
import argparse
import base64
import hashlib
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()


class ZoomDownloader:
    """
    Download and manage Zoom cloud recordings for given user emails.

    Supports comprehensive date-range searches, caching of processed meetings,
    and optional diagnostic output behind flags.
    """
    def __init__(self, downloads_dir="downloads", metadata_dir="metadata", verbose=False):
        self.client_id = os.getenv("ZOOM_CLIENT_ID")
        self.client_secret = os.getenv("ZOOM_CLIENT_SECRET")
        self.account_id = os.getenv("ZOOM_ACCOUNT_ID")
        self.api_base = "https://api.zoom.us/v2"

        self.downloads_dir = Path(downloads_dir)
        self.metadata_dir = Path(metadata_dir)
        self.downloads_dir.mkdir(exist_ok=True)
        self.metadata_dir.mkdir(exist_ok=True)

        self.processed_file = self.metadata_dir / "processed_meetings.json"
        self.processed_meetings = self.load_processed_meetings()
        self.verbose = verbose

    def load_processed_meetings(self):
        if self.processed_file.exists():
            with open(self.processed_file, 'r') as f:
                return set(json.load(f))
        return set()

    def save_processed_meetings(self):
        with open(self.processed_file, 'w') as f:
            json.dump(list(self.processed_meetings), f)

    def show_processed_summary(self):
        """Print a summary of already processed meetings."""
        if not self.processed_meetings:
            print("📋 No meetings processed yet")
            return

        latest = max(self.metadata_dir.glob("downloads_*.json"), default=None, key=lambda p: p.stat().st_mtime)
        print(f"📋 Already processed {len(self.processed_meetings)} recordings")
        if self.verbose and latest:
            with open(latest) as f:
                data = json.load(f)
            by_date = defaultdict(list)
            for item in data:
                key = item.get('date_key', item.get('start_time','')[:10])
                by_date[key].append(item)
            for date in sorted(by_date):
                topics = {i['topic'] for i in by_date[date]}
                print(f"  {date}: {len(topics)} meetings, {len(by_date[date])} files")

    def clear_processed_cache(self):
        """Force re-download by clearing cache."""
        if self.processed_file.exists():
            backup = self.metadata_dir / f"processed_meetings_backup_{datetime.now():%Y%m%d_%H%M%S}.json"
            self.processed_file.rename(backup)
            print(f"🔄 Cleared cache (backup: {backup.name})")
        self.processed_meetings.clear()

    def get_zoom_token(self):
        """Obtain OAuth access token for Zoom API."""
        creds = f"{self.client_id}:{self.client_secret}"
        encoded = base64.b64encode(creds.encode()).decode()
        headers = {"Content-Type": "application/x-www-form-urlencoded",
                   "Authorization": f"Basic {encoded}"}
        data = {"grant_type": "account_credentials", "account_id": self.account_id}
        resp = requests.post("https://zoom.us/oauth/token", headers=headers, data=data)
        resp.raise_for_status()
        return resp.json()["access_token"]

    def get_user_id(self, email, token):
        url = f"{self.api_base}/users/{email}"
        headers = {"Authorization": f"Bearer {token}"}
        r = requests.get(url, headers=headers)
        r.raise_for_status()
        return r.json()["id"]

    def create_meeting_key(self, topic, start_time, meeting_id, rec_type, file_type):
        clean = "".join(c.lower() for c in topic if c.isalnum() or c.isspace()).strip()
        date = start_time[:10] if start_time else "unknown"
        key = f"{clean}_{date}_{meeting_id}_{rec_type}_{file_type}"
        return hashlib.md5(key.encode()).hexdigest()[:16]

    def download_recording(self, url, path):
        if self.verbose:
            print(f"📥 Downloading {path.name}")
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        with open(path, 'wb') as f:
            for chunk in resp.iter_content(8192):
                if chunk:
                    f.write(chunk)
        return True

    def get_all_recordings(self, email):
        token = self.get_zoom_token()
        user_id = self.get_user_id(email, token)
        headers = {"Authorization": f"Bearer {token}"}
        all_recs = []
        found = set()
        ranges = [
            ("2018-01-01","2019-12-31"),("2020-01-01","2020-12-31"),
            ("2021-01-01","2021-12-31"),("2022-01-01","2022-12-31"),
            ("2023-01-01","2023-12-31"),("2024-01-01","2024-12-31"),
            ("2025-01-01","2025-12-31"),
        ]
        base = f"{self.api_base}/users/{user_id}/recordings"
        for start, end in ranges:
            if self.verbose:
                print(f"🔍 Searching {start} to {end}")
            params = {"page_size":300, "from":start, "to":end}
            while True:
                r = requests.get(base, headers=headers, params=params)
                if not r.ok:
                    if r.status_code==429:
                        time.sleep(30); continue
                    break
                data = r.json()
                for m in data.get('meetings',[]):
                    mid = m.get('id')
                    if mid in found: continue
                    found.add(mid)
                    files = []
                    for rf in m.get('recording_files',[]):
                        files.append({
                            'recording_type': rf.get('recording_type'),
                            'file_type': rf.get('file_type'),
                            'file_size': rf.get('file_size',0),
                            'download_url': rf.get('download_url')+f"?access_token={token}" if rf.get('download_url') else None,
                        })
                    if files:
                        all_recs.append({
                            'user_email': email,
                            'meeting_id': mid,
                            'topic': m.get('topic'),
                            'start_time': m.get('start_time'),
                            'duration': m.get('duration'),
                            'recording_files': files
                        })
                token_next = data.get('next_page_token')
                if token_next:
                    params['next_page_token'] = token_next
                    continue
                break
        return all_recs

    def process_recordings(self, emails):
        if isinstance(emails, str): emails = [emails]
        all_meta = []
        for email in emails:
            if self.verbose:
                print(f"\n➡️ Processing {email}")
            recs = self.get_all_recordings(email)
            for rec in recs:
                date = rec['start_time'][:10]
                safe = "".join(c for c in rec['topic'] if c.isalnum() or c in (' ','_')).strip()[:50]
                folder = self.downloads_dir / f"{date}_{safe}_{rec['meeting_id']}"
                folder.mkdir(exist_ok=True)
                for rf in rec['recording_files']:
                    key = self.create_meeting_key(rec['topic'], rec['start_time'], rec['meeting_id'], rf['recording_type'], rf['file_type'])
                    if key in self.processed_meetings: continue
                    if not rf['download_url']: continue
                    ext = rf['file_type'].lower()
                    fname = f"{rf['recording_type']}.{ext}"
                    path = folder / fname
                    if self.download_recording(rf['download_url'], path):
                        self.processed_meetings.add(key)
                        meta = {
                            'meeting_key': key,
                            'user_email': email,
                            'meeting_id': rec['meeting_id'],
                            'topic': rec['topic'],
                            'start_time': rec['start_time'],
                            'recording_type': rf['recording_type'],
                            'file_type': rf['file_type'],
                            'file_size': rf['file_size'],
                            'local_path': str(path),
                            'downloaded_at': datetime.now().isoformat(),
                            'date_key': date
                        }
                        all_meta.append(meta)
                        self.save_processed_meetings()
        out = self.metadata_dir / f"downloads_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(out,'w') as f: json.dump(all_meta, f, indent=2)
        print(f"✅ Downloaded {len(all_meta)} files → {out}")
        return all_meta


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Zoom Recording Downloader")
    p.add_argument('--emails', nargs='+', default=["cordobahouseschool3@gmail.com"], help="Zoom user emails")
    p.add_argument('--show-processed', action='store_true', help="Show processed summary")
    p.add_argument('--clear-cache', action='store_true', help="Clear cache")
    p.add_argument('--verbose', '-v', action='store_true', help="Verbose output")
    args = p.parse_args()

    dl = ZoomDownloader(verbose=args.verbose)
    if args.clear_cache:
        dl.clear_processed_cache()
        exit()
    if args.show_processed:
        dl.show_processed_summary()
        exit()

    dl.process_recordings(args.emails)
