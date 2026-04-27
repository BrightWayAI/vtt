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
import json
from pathlib import Path
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
HERE = Path(__file__).parent
TOKEN_OUT = HERE / "token.json"

# Use client_secret*.json from Desktop if present, otherwise reuse the
# client_id/client_secret already embedded in an existing token.json.
client_secret_file = next((Path.home() / "Desktop").glob("client_secret*.json"), None)

if client_secret_file:
    flow = InstalledAppFlow.from_client_secrets_file(str(client_secret_file), SCOPES)
elif TOKEN_OUT.exists():
    existing = json.loads(TOKEN_OUT.read_text())
    client_config = {
        "installed": {
            "client_id": existing["client_id"],
            "client_secret": existing["client_secret"],
            "redirect_uris": ["http://localhost"],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    }
    flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
else:
    raise SystemExit(
        "No client_secret*.json on Desktop and no existing token.json.\n"
        "Download the OAuth client JSON from Google Cloud Console > APIs & Services > Clients."
    )

creds = flow.run_local_server(port=0)
TOKEN_OUT.write_text(creds.to_json())
print(f"Saved token to {TOKEN_OUT}")
print(f"\nSet this env var:\n  GOOGLE_OAUTH_TOKEN={TOKEN_OUT}")
