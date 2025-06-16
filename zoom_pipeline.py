import os
import base64
import requests
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ─── Configuration from environment ────────────────────────────────────────────
ZOOM_CLIENT_ID     = os.getenv("ZOOM_CLIENT_ID")
ZOOM_CLIENT_SECRET = os.getenv("ZOOM_CLIENT_SECRET")
ZOOM_ACCOUNT_ID    = os.getenv("ZOOM_ACCOUNT_ID")  # Your Zoom account (organization) ID
ZOOM_API_BASE      = "https://api.zoom.us/v2"

def check_env_vars():
    if not all([ZOOM_CLIENT_ID, ZOOM_CLIENT_SECRET, ZOOM_ACCOUNT_ID]):
        print(f"ZOOM_CLIENT_ID: {ZOOM_CLIENT_ID}")
        print(f"ZOOM_CLIENT_SECRET: {ZOOM_CLIENT_SECRET}")
        print(f"ZOOM_ACCOUNT_ID: {ZOOM_ACCOUNT_ID}")
        raise RuntimeError(
            "Please set ZOOM_CLIENT_ID, ZOOM_CLIENT_SECRET, and ZOOM_ACCOUNT_ID in your .env file"
        )

# ─── 1. Generate an access token via Server-to-Server OAuth ────────────────────
def get_zoom_token() -> str:
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

# ─── 2. List recordings for a given Zoom user ID ───────────────────────────────
def list_recordings(user_id: str, token: str) -> List[Dict]:
    url     = f"{ZOOM_API_BASE}/users/{user_id}/recordings"
    headers = {"Authorization": f"Bearer {token}"}
    params  = {"page_size": 100}
    meetings: List[Dict] = []

    while True:
        r = requests.get(url, headers=headers, params=params)
        if not r.ok:
            print(f"Error {r.status_code} for {user_id}: {r.json()}")
            r.raise_for_status()
        payload = r.json()
        meetings.extend(payload.get("meetings", []))
        nxt = payload.get("next_page_token")
        if not nxt:
            break
        params["next_page_token"] = nxt

    return meetings

# ─── 3. Extract audio-only download link ────────────────────────────────────────
def get_audio_link(meeting: Dict, token: str) -> Optional[str]:
    for f in meeting.get("recording_files", []):
        if f.get("recording_type") == "audio_only":
            return f"{f['download_url']}?access_token={token}"
    return None

# ─── 4. (Optional) Resolve email→Zoom userID if needed ─────────────────────────
def get_user_id(email: str, token: str) -> str:
    url     = f"{ZOOM_API_BASE}/users/{email}"
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    return r.json()["id"]

def list_all_users(token: str) -> List[Dict]:
    """List all users in the Zoom account to help find the correct user ID"""
    url = f"{ZOOM_API_BASE}/users"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"page_size": 100, "status": "active"}
    users = []
    
    while True:
        r = requests.get(url, headers=headers, params=params)
        if not r.ok:
            print(f"Error {r.status_code} listing users: {r.json()}")
            r.raise_for_status()
        payload = r.json()
        users.extend(payload.get("users", []))
        nxt = payload.get("next_page_token")
        if not nxt:
            break
        params["next_page_token"] = nxt
    
    return users

def get_teachers(token: str) -> List[str]:
    # Option 1: If you have teacher emails, use them directly
    teacher_emails = [
        # "teacher1@example.com",
        # "teacher2@example.com",
    ]
    
    if teacher_emails:
        return [get_user_id(email, token) for email in teacher_emails]
    
    # Option 2: List all users to find the right ones
    print("Listing all users in your Zoom account:")
    users = list_all_users(token)
    for user in users:
        print(f"ID: {user['id']}, Email: {user['email']}, Name: {user.get('first_name', '')} {user.get('last_name', '')}")
    
    # For now, return empty list - you'll need to update this with actual user IDs
    print("\nPlease update the get_teachers() function with the correct user IDs or emails from the list above.")
    return []

def get_all_recordings(teachers: List[str], token: str) -> List[Dict]:
    all_recordings = []
    for user_id in teachers:
        meetings = list_recordings(user_id, token)
        for m in meetings:
            audio = get_audio_link(m, token)
            if not audio:
                continue
            all_recordings.append({
                "user_id":    user_id,
                "meeting_id": m["id"],
                "topic":      m.get("topic"),
                "start_time": m.get("start_time"),
                "audio_url":  audio
            })
    return all_recordings

def print_recordings(recordings: List[Dict]):
    print(f"Retrieved {len(recordings)} audio recordings")
    for r in recordings:
        print(r)

# ─── Main pipeline ─────────────────────────────────────────────────────────────
def main():
    check_env_vars()
    token = get_zoom_token()
    print("Successfully obtained access token!")
    
    teachers = get_teachers(token)
    
    if not teachers:
        print("No teachers configured. Please update the get_teachers() function with correct user IDs or emails.")
        return
    
    all_recordings = get_all_recordings(teachers, token)
    print_recordings(all_recordings)

if __name__ == "__main__":
    main()