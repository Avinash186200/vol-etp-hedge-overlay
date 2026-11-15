"""
drive_upload.py — push today's signal artifacts to Google Drive.

Files uploaded:
  output/signal_YYYYMMDD.txt
  output/signal_YYYYMMDD.json
  output/signal_latest.txt
  output/signal_latest.json
  history/score_history.csv

Drive folder ID is read from config.json (key: "drive_folder_id").
OAuth credentials in credentials.json + token.json (Google's standard).

Exit codes:
  0  uploaded successfully
  3  Drive failure (local copy still retained)
  4  config missing (uploads skipped)
"""
from __future__ import annotations
import sys, json, datetime as dt
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT  = HERE / "output"
HIST = HERE / "history"
LOG  = OUT / "drive_log.txt"
CONFIG_JSON = HERE / "config.json"
CRED = HERE / "credentials.json"
TOKEN = HERE / "token.json"

def _log(msg, mode="a"):
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    with open(LOG, mode, encoding="utf-8") as f: f.write(line + "\n")
    try: print(line, flush=True)
    except Exception: pass

def main() -> int:
    _log("=== drive_upload start ===", mode="w")
    if not CONFIG_JSON.exists():
        _log("config.json missing — skipping Drive upload (local copy retained)"); return 4
    cfg = json.loads(CONFIG_JSON.read_text(encoding="utf-8"))
    folder_id = cfg.get("drive_folder_id")
    if not folder_id:
        _log("config.drive_folder_id not set — skipping upload"); return 4
    if not CRED.exists():
        _log("credentials.json missing — skipping (drop your Google OAuth creds here)"); return 4

    try:
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload
    except ImportError:
        _log("google-api libs not installed. pip install google-auth google-auth-oauthlib google-api-python-client")
        return 3

    SCOPES = ["https://www.googleapis.com/auth/drive.file"]
    creds = None
    if TOKEN.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(CRED), SCOPES)
            creds = flow.run_local_server(port=0)
        TOKEN.write_text(creds.to_json(), encoding="utf-8")

    service = build("drive", "v3", credentials=creds, cache_discovery=False)

    today_str = dt.datetime.now().strftime("%Y%m%d")
    candidates = [
        OUT  / f"signal_{today_str}.txt",
        OUT  / f"signal_{today_str}.json",
        OUT  / "signal_latest.txt",
        OUT  / "signal_latest.json",
        HIST / "score_history.csv",
    ]
    targets = [p for p in candidates if p.exists()]
    if not targets:
        _log("no files to upload (signal step likely failed)"); return 3

    uploaded = 0
    for p in targets:
        try:
            q = (f"name='{p.name}' and '{folder_id}' in parents and trashed=false")
            existing = service.files().list(q=q, fields="files(id,name)").execute().get("files", [])
            media = MediaFileUpload(str(p), resumable=False)
            if existing:
                fid = existing[0]["id"]
                service.files().update(fileId=fid, media_body=media).execute()
                _log(f"  updated {p.name} (id={fid})")
            else:
                meta = {"name": p.name, "parents": [folder_id]}
                f = service.files().create(body=meta, media_body=media, fields="id").execute()
                _log(f"  created {p.name} (id={f['id']})")
            uploaded += 1
        except Exception as e:
            _log(f"  upload {p.name} FAILED: {e}")

    if uploaded == 0:
        _log("all uploads failed"); return 3
    _log(f"=== drive_upload OK: {uploaded}/{len(targets)} files ==="); return 0

if __name__ == "__main__":
    sys.exit(main())
