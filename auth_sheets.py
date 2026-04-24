"""
Run this once to authorize the VTT app to write to Google Sheets.

Usage:
    python3 auth_sheets.py

It will open a browser tab. Sign in with the Google account that has
Editor access to the upload sheet and click Allow. A token.json file
will be saved next to this script.

Then set the env var on your server:
    GOOGLE_OAUTH_TOKEN=/path/to/VTT/token.json
"""
import os
from pathlib import Path
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
CLIENT_SECRET = next((Path.home() / "Desktop").glob("client_secret*.json"), None)
if CLIENT_SECRET is None:
    raise SystemExit("No client_secret*.json found on Desktop")
TOKEN_OUT = Path(__file__).parent / "token.json"


flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRET), SCOPES)
creds = flow.run_local_server(port=0)

TOKEN_OUT.write_text(creds.to_json())
print(f"Saved token to {TOKEN_OUT}")
print(f"\nSet this env var:\n  GOOGLE_OAUTH_TOKEN={TOKEN_OUT}")
