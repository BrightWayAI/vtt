import csv
import io
import json
import logging
import os
import re
import subprocess
import tempfile
from datetime import date
from pathlib import Path
from urllib.parse import unquote, urlparse

import httpx
import psycopg2
from fastapi import FastAPI, File, Form, Request, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from pydub import AudioSegment

from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

MAX_UPLOAD_SIZE = 500 * 1024 * 1024  # 500 MB for direct uploads
CHUNK_DURATION_MS = 10 * 60 * 1000   # 10 minutes per Whisper chunk

AIRTABLE_TOKEN = os.environ.get("AIRTABLE_TOKEN", "")
AIRTABLE_BASE_ID = "appf82sOr6qFvVj6z"
AIRTABLE_TASKS_TABLE = "tblnMlOiI3q3Zj4jo"

VIDEOS_UPLOAD_DIR = Path(os.path.expanduser(
    os.environ.get("VIDEOS_UPLOAD_DIR", "~/Desktop/Videos for Upload")
))

GOOGLE_OAUTH_TOKEN = os.environ.get("GOOGLE_OAUTH_TOKEN", "")
UPLOAD_SHEET_ID = "1OUQ0NyIaCOsPvYUCQhJ41roINv9fT9wrV4llHl74BpY"
UPLOAD_SHEET_TAB = "Information for Video Uploading"

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def get_db():
    url = os.environ.get("DATABASE_URL")
    if not url:
        return None
    return psycopg2.connect(url)


def init_db():
    conn = get_db()
    if conn is None:
        return
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS transcriptions (
                    id          SERIAL PRIMARY KEY,
                    filename    TEXT NOT NULL,
                    file_size   INTEGER NOT NULL,
                    duration_s  REAL,
                    vtt_content TEXT NOT NULL,
                    created_at  TIMESTAMP DEFAULT NOW(),
                    ip_address  TEXT
                );
            """)
        conn.commit()
    finally:
        conn.close()


def parse_vtt_duration(vtt_text: str) -> float | None:
    matches = re.findall(r"(\d{2}):(\d{2}):(\d{2})\.(\d{3})", vtt_text)
    if not matches:
        return None
    h, m, s, ms = matches[-1]
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def save_transcription(filename: str, file_size: int, vtt_content: str, ip: str | None) -> int | None:
    conn = get_db()
    if conn is None:
        return None
    try:
        duration = parse_vtt_duration(vtt_content)
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO transcriptions (filename, file_size, duration_s, vtt_content, ip_address)
                   VALUES (%s, %s, %s, %s, %s) RETURNING id""",
                (filename, file_size, duration, vtt_content, ip),
            )
            row = cur.fetchone()
        conn.commit()
        return row[0] if row else None
    finally:
        conn.close()


@app.on_event("startup")
def startup():
    init_db()


# ---------------------------------------------------------------------------
# Filename helpers
# ---------------------------------------------------------------------------

def extract_video_id(filename: str) -> int | None:
    """Parse leading Video ID from filenames like '121_What is a Community_Final.mp4'."""
    m = re.match(r'^(\d+)[_\s]', Path(filename).stem)
    return int(m.group(1)) if m else None


def save_to_upload_folder(filename: str, vtt_content: str) -> None:
    """Write VTT to ~/Desktop/Videos for Upload if the folder exists."""
    if VIDEOS_UPLOAD_DIR.is_dir():
        out = VIDEOS_UPLOAD_DIR / (Path(filename).stem + ".vtt")
        out.write_text(vtt_content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Airtable / storyboard helpers
# ---------------------------------------------------------------------------

_VO_LINE_RE = re.compile(r'(?:Narrator|Dialogue)[^:]*:\s*[""]?(.+)', re.IGNORECASE)


def fetch_airtable_record(video_id: int) -> tuple[str, str, str] | None:
    """Fetch (record_id, topic, storyboard_url) for a Video ID. Returns None if unavailable."""
    if not AIRTABLE_TOKEN:
        return None
    try:
        with httpx.Client(timeout=15) as client:
            resp = client.get(
                f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TASKS_TABLE}",
                params={
                    "filterByFormula": f"{{Video ID #}}={video_id}",
                    "fields[]": ["Video Topic", "Storyboard Link"],
                    "maxRecords": "1",
                },
                headers={"Authorization": f"Bearer {AIRTABLE_TOKEN}"},
            )
            resp.raise_for_status()
        records = resp.json().get("records", [])
        if not records:
            return None
        rec = records[0]
        record_id = rec["id"]
        fields = rec.get("fields", {})
        topic = (fields.get("Video Topic") or "").strip()
        storyboard_url = fields.get("Storyboard Link", "")
        return record_id, topic, storyboard_url
    except Exception:
        return None


def fetch_storyboard_vo(storyboard_url: str) -> str | None:
    """Parse a Google Doc storyboard URL and return the VO text for use as a Whisper prompt."""
    if not storyboard_url:
        return None
    try:
        m = re.search(r"/document/d/([a-zA-Z0-9_-]+)", storyboard_url)
        if not m:
            return None
        with httpx.Client(follow_redirects=True, timeout=30) as client:
            resp = client.get(
                f"https://docs.google.com/document/d/{m.group(1)}/export?format=txt"
            )
            resp.raise_for_status()
            text = resp.text
        vo_lines = []
        for line in text.splitlines():
            line = line.strip()
            lm = _VO_LINE_RE.match(line)
            if lm:
                content = lm.group(1).strip().strip('"').strip('"').strip('"')
                if content and not content.startswith("[") and len(content) > 3:
                    vo_lines.append(content)
        # Whisper's prompt window is ~224 tokens; 900 chars is a safe limit
        return " ".join(vo_lines)[:900] if vo_lines else None
    except Exception:
        return None


def generate_thumbnail(file_path: str, stem: str) -> None:
    """Extract a frame from the video at ~10 s and save a 552x414 PNG to the upload folder."""
    if not VIDEOS_UPLOAD_DIR.is_dir():
        logger.warning("generate_thumbnail: VIDEOS_UPLOAD_DIR does not exist: %s", VIDEOS_UPLOAD_DIR)
        return
    out_path = VIDEOS_UPLOAD_DIR / f"{stem}.png"
    if out_path.exists():
        return
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", "10",
                "-i", file_path,
                "-vframes", "1",
                "-vf", "scale=552:414:force_original_aspect_ratio=increase,crop=552:414",
                "-q:v", "2",
                str(out_path),
            ],
            capture_output=True,
            check=True,
        )
        logger.info("generate_thumbnail: saved %s", out_path)
    except Exception as exc:
        logger.error("generate_thumbnail failed: %s", exc)


def update_upload_date(record_id: str) -> None:
    """Set Video Assets Uploaded to today's date on the Airtable record."""
    if not AIRTABLE_TOKEN or not record_id:
        return
    try:
        with httpx.Client(timeout=15) as client:
            client.patch(
                f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TASKS_TABLE}/{record_id}",
                json={"fields": {"Video Assets Uploaded": date.today().isoformat()}},
                headers={
                    "Authorization": f"Bearer {AIRTABLE_TOKEN}",
                    "Content-Type": "application/json",
                },
            )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Master sheet / upload CSV helpers
# ---------------------------------------------------------------------------

_MASTER_SHEET_CSV_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1h8eckJjQIHForqxywqqEhNJ_3FdE2_fZe66NIUjR5yA"
    "/export?format=csv&gid=1118701471"
)

_MASTER_SHEET_COLS = [
    "Video ID #",
    "Category",
    "Topic (Video Name)",
    "Grade Level",
    "Standard",
    "Key Vocab",
    "Learning Objective",
    "Topic Question 1 - Freeze Question",
    "Topic Question 2 - Reflection Ending Question",
    "Conversation Starter 1",
    "Conversation Starter 2",
]

_UPLOAD_CSV_COLS = [
    "Video ID #",
    "Sponsor",
    "Category",
    "Collection",
    "Topic (Video Name)",
    "Video Available for Demo Mode",
    "Grade Level",
    "Standard",
    "Key Vocab",
    "Learning Objective",
    "Topic Question 1 - Freeze Question",
    "Topic Question 2 - Reflection Ending Question",
    "Conversation Starter 1",
    "Conversation Starter 2",
]

_SEL_QUESTION_FIELDS = [
    "Topic Question 1 - Freeze Question",
    "Topic Question 2 - Reflection Ending Question",
    "Conversation Starter 1",
    "Conversation Starter 2",
]

_SEL_WORDS_RE = re.compile(
    r"\b(brave|bravery|courageous|courage|heroic|unfair|injustice|"
    r"amazing|incredible|wonderful|inspiring|inspire|hopeful|hope|"
    r"determined|determination|resilient|resilience|proud|joyful|"
    r"dignity|agency|empowerment|empower)\b",
    re.IGNORECASE,
)

_SEL_PHRASES_RE = re.compile(
    r"how do you think (?:he|she|they|[a-z]+) felt"
    r"|how did it feel to"
    r"|imagine how (?:hard|scary|exciting)"
    r"|what do you feel when"
    r"|how do you feel about"
    r"|what would it feel like"
    r"|how (?:do you think you'?d?|would you) feel"
    r"|what (?:made|makes) \w+ (?:brave|strong|courageous|determined)"
    r"|how did \w+ show"
    r"|what qualities made"
    r"|what would you do if"
    r"|if you were .+, how would you feel"
    r"|why is (?:sharing|kindness|fairness|helping) important"
    r"|what does this teach us about",
    re.IGNORECASE,
)


def _has_sel_violation(text: str) -> bool:
    return bool(_SEL_WORDS_RE.search(text) or _SEL_PHRASES_RE.search(text))


def fetch_master_sheet_row(video_id: int) -> dict | None:
    """Fetch the master Google Sheet CSV and return the row matching video_id."""
    try:
        with httpx.Client(follow_redirects=True, timeout=30) as client:
            resp = client.get(_MASTER_SHEET_CSV_URL)
            resp.raise_for_status()
        reader = csv.DictReader(io.StringIO(resp.text))
        for row in reader:
            raw_id = row.get("Video ID #", "").strip()
            try:
                if int(raw_id) == video_id:
                    logger.info("fetch_master_sheet_row: found Video ID %s", video_id)
                    return {col: row.get(col, "").strip() for col in _MASTER_SHEET_COLS}
            except (ValueError, TypeError):
                continue
    except Exception as exc:
        logger.error("fetch_master_sheet_row failed: %s", exc)
    logger.warning("fetch_master_sheet_row: no row found for Video ID %s", video_id)
    return None


def sel_check_and_rewrite(row: dict) -> dict:
    """Check question fields for SEL violations and auto-rewrite via OpenAI."""
    row = dict(row)
    violations = [f for f in _SEL_QUESTION_FIELDS if _has_sel_violation(row.get(f, ""))]
    if not violations:
        return row

    client = OpenAI()
    topic = row.get("Topic (Video Name)", "this video")
    objective = row.get("Learning Objective", "")
    vocab = row.get("Key Vocab", "")

    for field in violations:
        original = row[field]
        is_freeze = "Freeze Question" in field
        rewrite_instruction = (
            "Rewrite as a factual comprehension question anchored to episode content. "
            "Use What/Who/When/Where/How framing. No emotion, no speculation, no moralizing. "
            "Return only the rewritten question, nothing else."
            if is_freeze else
            "Rewrite to anchor in facts or concrete actions instead of feelings or traits. "
            "Keep the subject matter. Return only the rewritten question, nothing else."
        )
        prompt = (
            f"You are editing a question for an educational video titled '{topic}'.\n"
            f"Learning objective: {objective}\n"
            f"Key vocab: {vocab}\n\n"
            f"The question below violates SEL (Social-Emotional Learning) guidelines "
            f"because it uses feelings-speculation, character-trait framing, moralizing, "
            f"or forbidden SEL words (brave, courage, inspiring, hope, etc.).\n\n"
            f"Original question: {original}\n\n"
            f"{rewrite_instruction}"
        )
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.3,
            )
            rewritten = response.choices[0].message.content.strip()
            if rewritten:
                row[field] = rewritten
        except Exception:
            pass

    return row


def _write_upload_csv_fallback(row: dict) -> None:
    """Fallback: append row to a local CSV when the Sheets webhook is unavailable."""
    if not VIDEOS_UPLOAD_DIR.is_dir():
        return
    csv_path = VIDEOS_UPLOAD_DIR / f"video-upload-rows-{date.today().isoformat()}.csv"
    file_exists = csv_path.exists()
    try:
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_UPLOAD_CSV_COLS, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
            writer.writerow({col: row.get(col, "") for col in _UPLOAD_CSV_COLS})
    except Exception:
        pass


def write_to_upload_sheet(row: dict) -> None:
    """Append the row to the Google Sheet via OAuth2; falls back to local CSV."""
    import gspread
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request

    values = [row.get(col, "") for col in _UPLOAD_CSV_COLS]
    if not GOOGLE_OAUTH_TOKEN:
        logger.warning("write_to_upload_sheet: GOOGLE_OAUTH_TOKEN not set, falling back to CSV")
        _write_upload_csv_fallback(row)
        return
    try:
        # Env var can be raw JSON (Railway) or a file path (local)
        if GOOGLE_OAUTH_TOKEN.strip().startswith("{"):
            info = json.loads(GOOGLE_OAUTH_TOKEN)
            creds = Credentials.from_authorized_user_info(
                info,
                scopes=["https://www.googleapis.com/auth/spreadsheets"],
            )
        else:
            creds = Credentials.from_authorized_user_file(
                GOOGLE_OAUTH_TOKEN,
                scopes=["https://www.googleapis.com/auth/spreadsheets"],
            )
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
        gc = gspread.authorize(creds)
        ws = gc.open_by_key(UPLOAD_SHEET_ID).worksheet(UPLOAD_SHEET_TAB)
        ws.append_row(values, value_input_option="USER_ENTERED")
        logger.info("write_to_upload_sheet: appended row for Video ID %s", row.get("Video ID #"))
    except Exception as exc:
        logger.error("write_to_upload_sheet failed: %s", exc, exc_info=True)
        _write_upload_csv_fallback(row)


# ---------------------------------------------------------------------------
# Transcription helpers
# ---------------------------------------------------------------------------

# Riptoes is consistently misheared in the opening "where'd you go this time X".
# This regex catches the phrase regardless of what word Whisper used for the name.
_RIPTOES_RE = re.compile(
    r"where'?d?\s+(?:did\s+)?you\s+go\s+this\s+time\s+(\w+)",
    re.IGNORECASE,
)


def apply_context_corrections(words: list[dict], segments: list[dict]) -> list[dict]:
    """Fix proper nouns that are deterministic from their surrounding context."""
    words = [w.copy() for w in words]
    for seg in segments:
        m = _RIPTOES_RE.search(seg.get("text", ""))
        if not m:
            continue
        bad = m.group(1).strip().lower()
        for w in words:
            if (
                w["start"] >= seg["start"] - 0.05
                and w["end"] <= seg["end"] + 0.05
                and w["word"].strip().lower() == bad
            ):
                w["word"] = "Riptoes"
                break
    return words


def fmt_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds % 1) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def transcribe_chunk(client: OpenAI, chunk_path: str, prompt: str | None = None):
    with open(chunk_path, "rb") as f:
        kwargs = dict(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"],
        )
        if prompt:
            kwargs["prompt"] = prompt
        return client.audio.transcriptions.create(**kwargs)


def build_word_highlight_vtt(all_words, all_segments) -> str:
    if not all_words:
        return "WEBVTT\nKind: captions\nLanguage: en\n\n"

    lines = ["WEBVTT", "Kind: captions", "Language: en", ""]

    for seg in all_segments:
        seg_words = [
            w for w in all_words
            if w["start"] >= seg["start"] - 0.01 and w["end"] <= seg["end"] + 0.01
        ]
        if not seg_words:
            continue

        for i, word in enumerate(seg_words):
            start = fmt_ts(word["start"])
            end = fmt_ts(word["end"])

            parts = []
            for j, w in enumerate(seg_words):
                txt = w["word"].strip()
                if j == i:
                    parts.append(f"<v>{txt}</v>")
                else:
                    parts.append(txt)

            lines.append(f"{start} --> {end}")
            lines.append(" ".join(parts))
            lines.append("")

    return "\n".join(lines)


def transcribe_file(file_path: str, vo_prompt: str | None = None) -> str:
    client = OpenAI()
    audio = AudioSegment.from_file(file_path)
    duration_ms = len(audio)

    if duration_ms <= CHUNK_DURATION_MS:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            audio.export(tmp.name, format="mp3", bitrate="64k")
            result = transcribe_chunk(client, tmp.name, prompt=vo_prompt)
            os.unlink(tmp.name)

        words = [{"word": w.word, "start": w.start, "end": w.end}
                 for w in (getattr(result, "words", None) or [])]
        segments = [{"start": s.start, "end": s.end, "text": s.text}
                    for s in (getattr(result, "segments", None) or [])]
        words = apply_context_corrections(words, segments)
        return build_word_highlight_vtt(words, segments)

    all_words = []
    all_segments = []

    for chunk_start_ms in range(0, duration_ms, CHUNK_DURATION_MS):
        chunk_end_ms = min(chunk_start_ms + CHUNK_DURATION_MS, duration_ms)
        chunk = audio[chunk_start_ms:chunk_end_ms]
        offset_s = chunk_start_ms / 1000.0

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            chunk.export(tmp.name, format="mp3", bitrate="64k")
            result = transcribe_chunk(client, tmp.name, prompt=vo_prompt)
            os.unlink(tmp.name)

        for w in getattr(result, "words", None) or []:
            all_words.append({
                "word": w.word,
                "start": w.start + offset_s,
                "end": w.end + offset_s,
            })

        for s in getattr(result, "segments", None) or []:
            all_segments.append({
                "start": s.start + offset_s,
                "end": s.end + offset_s,
                "text": s.text,
            })

    all_words = apply_context_corrections(all_words, all_segments)
    return build_word_highlight_vtt(all_words, all_segments)


CONTENT_TYPE_SUFFIX = {
    "video/mp4": ".mp4",
    "video/quicktime": ".mov",
    "video/x-msvideo": ".avi",
    "video/webm": ".webm",
    "video/x-matroska": ".mkv",
    "audio/mpeg": ".mp3",
    "audio/mp4": ".m4a",
    "audio/wav": ".wav",
    "audio/ogg": ".ogg",
    "audio/webm": ".webm",
    "audio/flac": ".flac",
    "audio/x-flac": ".flac",
}


def download_url_to_temp(url: str) -> tuple[str, str, int]:
    """Download a URL to a temp file. Returns (path, filename, size)."""
    tmp = None
    size = 0
    with httpx.stream("GET", url, follow_redirects=True, timeout=300) as r:
        r.raise_for_status()

        # Determine filename: prefer Content-Disposition, then final URL path
        content_disposition = r.headers.get("content-disposition", "")
        filename = None
        if content_disposition:
            m = re.search(r'filename\*?=["\']?(?:UTF-8\'\')?([^"\';]+)', content_disposition, re.IGNORECASE)
            if m:
                filename = unquote(m.group(1).strip())

        if not filename:
            final_url = str(r.url)
            parsed_final = urlparse(final_url)
            filename = Path(unquote(parsed_final.path)).name or "download"

        # Determine suffix: prefer filename extension, then Content-Type
        suffix = Path(filename).suffix
        if not suffix:
            content_type = r.headers.get("content-type", "").split(";")[0].strip()
            suffix = CONTENT_TYPE_SUFFIX.get(content_type, ".mp4")
            filename = filename + suffix

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        for chunk in r.iter_bytes(chunk_size=1024 * 64):
            tmp.write(chunk)
            size += len(chunk)
    tmp.close()
    return tmp.name, filename, size


# ---------------------------------------------------------------------------
# HTML pages
# ---------------------------------------------------------------------------

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>VTT Generator</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: #0f0f0f; color: #e0e0e0; display: flex; justify-content: center; align-items: center; min-height: 100vh; }
  .container { background: #1a1a1a; border-radius: 12px; padding: 2.5rem; max-width: 520px; width: 90%; box-shadow: 0 4px 24px rgba(0,0,0,0.4); }
  h1 { font-size: 1.5rem; margin-bottom: 0.5rem; }
  p.sub { color: #888; margin-bottom: 1.5rem; font-size: 0.9rem; }

  /* Tabs */
  .tabs { display: flex; gap: 0; margin-bottom: 1.5rem; border-bottom: 1px solid #333; }
  .tab { padding: 0.5rem 1rem; cursor: pointer; color: #888; font-size: 0.9rem; border-bottom: 2px solid transparent; transition: all 0.2s; }
  .tab:hover { color: #ccc; }
  .tab.active { color: #4a9eff; border-bottom-color: #4a9eff; }
  .tab-content { display: none; }
  .tab-content.active { display: block; }

  label.file-label { display: block; border: 2px dashed #333; border-radius: 8px; padding: 2rem; text-align: center; cursor: pointer; transition: border-color 0.2s; margin-bottom: 1rem; }
  label.file-label:hover { border-color: #555; }
  label.file-label.has-file { border-color: #4a9eff; }
  label.file-label.dragover { border-color: #4a9eff; background: rgba(74,158,255,0.05); }
  input[type="file"] { display: none; }
  .file-name { font-size: 0.85rem; color: #4a9eff; margin-top: 0.5rem; word-break: break-all; }

  input[type="text"], textarea { width: 100%; padding: 0.75rem; border: 1px solid #333; border-radius: 8px; background: #111; color: #e0e0e0; font-family: system-ui, sans-serif; font-size: 0.9rem; margin-bottom: 1rem; }
  input[type="text"]:focus, textarea:focus { outline: none; border-color: #4a9eff; }
  textarea { min-height: 120px; resize: vertical; }

  button { width: 100%; padding: 0.75rem; border: none; border-radius: 8px; background: #4a9eff; color: #fff; font-size: 1rem; cursor: pointer; transition: background 0.2s; }
  button:hover { background: #3a8eef; }
  button:disabled { background: #333; color: #666; cursor: not-allowed; }

  .status { margin-top: 1rem; font-size: 0.9rem; text-align: center; }
  .status.error { color: #ff6b6b; }
  .status.success a { color: #4a9eff; text-decoration: none; font-weight: 600; }

  .batch-log { margin-top: 1rem; font-size: 0.85rem; max-height: 300px; overflow-y: auto; }
  .batch-item { padding: 0.5rem 0; border-bottom: 1px solid #222; display: flex; justify-content: space-between; align-items: center; }
  .batch-item .name { color: #ccc; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; margin-right: 0.5rem; }
  .batch-item .state { font-size: 0.8rem; white-space: nowrap; }
  .batch-item .state.queued { color: #666; }
  .batch-item .state.downloading { color: #f0ad4e; }
  .batch-item .state.transcribing { color: #f0ad4e; }
  .batch-item .state.done { color: #5cb85c; }
  .batch-item .state.error { color: #ff6b6b; }
  .batch-item a { color: #4a9eff; text-decoration: none; font-size: 0.8rem; margin-left: 0.5rem; }

  .nav { margin-top: 1.5rem; text-align: center; }
  .nav a { color: #888; font-size: 0.85rem; text-decoration: none; }
  .nav a:hover { color: #4a9eff; }
</style>
</head>
<body>
<div class="container">
  <h1>VTT Generator</h1>
  <p class="sub">Generate word-highlighted subtitle files (.vtt)</p>

  <div class="tabs">
    <div class="tab active" data-tab="file">File Upload</div>
    <div class="tab" data-tab="url">URL</div>
    <div class="tab" data-tab="batch">Batch URLs</div>
  </div>

  <!-- File upload tab -->
  <div class="tab-content active" id="tab-file">
    <form id="file-form">
      <label class="file-label" id="drop-label" for="file-input">
        <span id="label-text">Click to select or drag a file here</span>
        <div class="file-name" id="file-name"></div>
        <input type="file" id="file-input" accept="video/*,audio/*,.mp3,.mp4,.m4a,.wav,.webm,.ogg,.flac,.mpeg,.mpga">
      </label>
      <button type="submit" id="file-btn" disabled>Generate VTT</button>
    </form>
    <div class="status" id="file-status"></div>
  </div>

  <!-- URL tab -->
  <div class="tab-content" id="tab-url">
    <form id="url-form">
      <input type="text" id="url-input" placeholder="Paste a direct link to a video or audio file">
      <button type="submit" id="url-btn">Generate VTT</button>
    </form>
    <div class="status" id="url-status"></div>
  </div>

  <!-- Batch tab -->
  <div class="tab-content" id="tab-batch">
    <form id="batch-form">
      <textarea id="batch-input" placeholder="Paste URLs, one per line"></textarea>
      <button type="submit" id="batch-btn">Process All</button>
    </form>
    <div class="batch-log" id="batch-log"></div>
  </div>

  <div class="nav"><a href="/history">View transcription history</a></div>
</div>
<script>
  // --- Tabs ---
  document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      tab.classList.add('active');
      document.getElementById('tab-' + tab.dataset.tab).classList.add('active');
    });
  });

  // --- File upload ---
  const fileForm = document.getElementById('file-form');
  const fileInput = document.getElementById('file-input');
  const fileNameEl = document.getElementById('file-name');
  const label = document.getElementById('drop-label');
  const fileBtn = document.getElementById('file-btn');
  const fileStatus = document.getElementById('file-status');

  function setFile(file) {
    const dt = new DataTransfer();
    dt.items.add(file);
    fileInput.files = dt.files;
    fileNameEl.textContent = file.name + ' (' + (file.size / 1024 / 1024).toFixed(1) + ' MB)';
    label.classList.add('has-file');
    fileBtn.disabled = false;
  }

  fileInput.addEventListener('change', () => { if (fileInput.files.length) setFile(fileInput.files[0]); });

  label.addEventListener('dragover', (e) => { e.preventDefault(); e.stopPropagation(); label.classList.add('dragover'); });
  label.addEventListener('dragleave', (e) => { e.preventDefault(); e.stopPropagation(); label.classList.remove('dragover'); });
  label.addEventListener('drop', (e) => {
    e.preventDefault(); e.stopPropagation(); label.classList.remove('dragover');
    if (e.dataTransfer.files.length) setFile(e.dataTransfer.files[0]);
  });

  fileForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const file = fileInput.files[0];
    if (!file) return;
    fileBtn.disabled = true;
    fileStatus.className = 'status';
    fileStatus.textContent = 'Transcribing… this may take a few minutes for large files.';
    try {
      const fd = new FormData();
      fd.append('file', file);
      const res = await fetch('/transcribe', { method: 'POST', body: fd });
      if (!res.ok) { const err = await res.json(); throw new Error(err.detail || 'Transcription failed'); }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const name = file.name.replace(/\\.[^.]+$/, '') + '.vtt';
      fileStatus.className = 'status success';
      fileStatus.innerHTML = '<a href="' + url + '" download="' + name + '">Download ' + name + '</a>';
    } catch (err) {
      fileStatus.className = 'status error';
      fileStatus.textContent = err.message;
    } finally { fileBtn.disabled = false; }
  });

  // --- URL ---
  const urlForm = document.getElementById('url-form');
  const urlInput = document.getElementById('url-input');
  const urlBtn = document.getElementById('url-btn');
  const urlStatus = document.getElementById('url-status');

  urlForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const url = urlInput.value.trim();
    if (!url) return;
    urlBtn.disabled = true;
    urlStatus.className = 'status';
    urlStatus.textContent = 'Downloading and transcribing…';
    try {
      const body = 'url=' + encodeURIComponent(url);
      const res = await fetch('/transcribe-url', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body,
      });
      if (!res.ok) { const err = await res.json(); throw new Error(err.detail || 'Failed'); }
      const blob = await res.blob();
      const disp = res.headers.get('Content-Disposition') || '';
      const match = disp.match(/filename="(.+?)"/);
      const name = match ? match[1] : 'subtitles.vtt';
      const blobUrl = URL.createObjectURL(blob);
      urlStatus.className = 'status success';
      urlStatus.innerHTML = '<a href="' + blobUrl + '" download="' + name + '">Download ' + name + '</a>';
    } catch (err) {
      urlStatus.className = 'status error';
      urlStatus.textContent = err.message;
    } finally { urlBtn.disabled = false; }
  });

  // --- Batch ---
  const batchForm = document.getElementById('batch-form');
  const batchInput = document.getElementById('batch-input');
  const batchBtn = document.getElementById('batch-btn');
  const batchLog = document.getElementById('batch-log');

  batchForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const urls = batchInput.value.trim().split('\\n').map(u => u.trim()).filter(Boolean);
    if (!urls.length) return;
    batchBtn.disabled = true;
    batchLog.innerHTML = '';

    // Build item rows
    const items = {};
    urls.forEach((url, i) => {
      const name = url.split('/').pop().split('?')[0] || ('file-' + (i+1));
      const div = document.createElement('div');
      div.className = 'batch-item';
      div.innerHTML = '<span class="name" title="' + url + '">' + name + '</span><span class="state queued" id="bs-' + i + '">queued</span>';
      batchLog.appendChild(div);
      items[i] = { url, name, el: div };
    });

    const es = new EventSource('/batch?urls=' + encodeURIComponent(JSON.stringify(urls)));
    es.onmessage = (evt) => {
      const d = JSON.parse(evt.data);
      const stateEl = document.getElementById('bs-' + d.index);
      if (!stateEl) return;
      const item = items[d.index];
      if (d.status === 'downloading') {
        stateEl.className = 'state downloading';
        stateEl.textContent = 'downloading…';
      } else if (d.status === 'transcribing') {
        stateEl.className = 'state transcribing';
        stateEl.textContent = 'transcribing…';
      } else if (d.status === 'done') {
        stateEl.className = 'state done';
        stateEl.innerHTML = 'done <a href="/download/' + d.id + '">download</a>';
      } else if (d.status === 'error') {
        stateEl.className = 'state error';
        stateEl.textContent = d.message || 'error';
      } else if (d.status === 'complete') {
        es.close();
        batchBtn.disabled = false;
      }
    };
    es.onerror = () => { es.close(); batchBtn.disabled = false; };
  });
</script>
</body>
</html>
"""

HISTORY_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Transcription History</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: #0f0f0f; color: #e0e0e0; display: flex; justify-content: center; padding: 2rem; }
  .container { background: #1a1a1a; border-radius: 12px; padding: 2.5rem; max-width: 720px; width: 100%; box-shadow: 0 4px 24px rgba(0,0,0,0.4); }
  h1 { font-size: 1.5rem; margin-bottom: 0.5rem; }
  .nav { margin-bottom: 1.5rem; }
  .nav a { color: #4a9eff; font-size: 0.85rem; text-decoration: none; }
  .nav a:hover { text-decoration: underline; }
  table { width: 100%; border-collapse: collapse; margin-top: 1rem; }
  th, td { text-align: left; padding: 0.6rem 0.75rem; border-bottom: 1px solid #2a2a2a; font-size: 0.9rem; }
  th { color: #888; font-weight: 600; }
  td a { color: #4a9eff; text-decoration: none; }
  td a:hover { text-decoration: underline; }
  .empty { color: #666; margin-top: 1.5rem; text-align: center; }
</style>
</head>
<body>
<div class="container">
  <div class="nav"><a href="/">&larr; Back to generator</a></div>
  <h1>Transcription History</h1>
  {{TABLE}}
</div>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


@app.post("/transcribe")
async def transcribe(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="File exceeds 500 MB limit.")

    original_name = file.filename or "unknown"
    stem = Path(original_name).stem
    video_id = extract_video_id(original_name)
    airtable = fetch_airtable_record(video_id) if video_id is not None else None
    record_id, topic, storyboard_url = airtable if airtable else (None, stem, "")
    vo_prompt = fetch_storyboard_vo(storyboard_url)

    suffix = Path(file.filename).suffix if file.filename else ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(contents)
        tmp.close()

        vtt_text = transcribe_file(tmp.name, vo_prompt=vo_prompt)

        ip = request.client.host if request.client else None
        save_transcription(original_name, len(contents), vtt_text, ip)
        save_to_upload_folder(original_name, vtt_text)
        generate_thumbnail(tmp.name, stem)
        update_upload_date(record_id)
        master_row = fetch_master_sheet_row(video_id) if video_id is not None else None
        if master_row:
            master_row = sel_check_and_rewrite(master_row)
            write_to_upload_sheet(master_row)

        out_name = stem + ".vtt"
        return Response(
            content=vtt_text,
            media_type="text/vtt",
            headers={"Content-Disposition": f'attachment; filename="{out_name}"'},
        )
    finally:
        os.unlink(tmp.name)


@app.post("/transcribe-url")
async def transcribe_url(request: Request, url: str = Form(...)):
    try:
        file_path, filename, file_size = download_url_to_temp(url)
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=400, detail=f"Download failed: HTTP {exc.response.status_code}")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Download failed: {exc}")

    stem = Path(filename).stem
    video_id = extract_video_id(filename)
    airtable = fetch_airtable_record(video_id) if video_id is not None else None
    record_id, topic, storyboard_url = airtable if airtable else (None, stem, "")
    vo_prompt = fetch_storyboard_vo(storyboard_url)

    try:
        vtt_text = transcribe_file(file_path, vo_prompt=vo_prompt)

        ip = request.client.host if request.client else None
        save_transcription(filename, file_size, vtt_text, ip)
        save_to_upload_folder(filename, vtt_text)
        generate_thumbnail(file_path, stem)
        update_upload_date(record_id)
        master_row = fetch_master_sheet_row(video_id) if video_id is not None else None
        if master_row:
            master_row = sel_check_and_rewrite(master_row)
            write_to_upload_sheet(master_row)

        out_name = stem + ".vtt"
        return Response(
            content=vtt_text,
            media_type="text/vtt",
            headers={"Content-Disposition": f'attachment; filename="{out_name}"'},
        )
    finally:
        os.unlink(file_path)


@app.get("/batch")
async def batch(request: Request, urls: str):
    """SSE endpoint: processes a list of URLs and streams progress events."""
    try:
        url_list = json.loads(urls)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid URL list.")

    if not isinstance(url_list, list) or len(url_list) == 0:
        raise HTTPException(status_code=400, detail="Provide at least one URL.")

    ip = request.client.host if request.client else None

    def event_stream():
        for i, url in enumerate(url_list):
            url = url.strip()
            if not url:
                continue
            try:
                yield f"data: {json.dumps({'index': i, 'status': 'downloading'})}\n\n"
                file_path, filename, file_size = download_url_to_temp(url)
            except Exception as exc:
                yield f"data: {json.dumps({'index': i, 'status': 'error', 'message': f'Download failed: {exc}'})}\n\n"
                continue

            try:
                yield f"data: {json.dumps({'index': i, 'status': 'transcribing'})}\n\n"
                fstem = Path(filename).stem
                vid = extract_video_id(filename)
                at = fetch_airtable_record(vid) if vid is not None else None
                b_record_id, b_topic, b_storyboard = at if at else (None, fstem, "")
                vo_prompt = fetch_storyboard_vo(b_storyboard)
                vtt_text = transcribe_file(file_path, vo_prompt=vo_prompt)
                tid = save_transcription(filename, file_size, vtt_text, ip)
                save_to_upload_folder(filename, vtt_text)
                generate_thumbnail(file_path, fstem)
                update_upload_date(b_record_id)
                b_master = fetch_master_sheet_row(vid) if vid is not None else None
                if b_master:
                    b_master = sel_check_and_rewrite(b_master)
                    write_to_upload_sheet(b_master)
                yield f"data: {json.dumps({'index': i, 'status': 'done', 'id': tid})}\n\n"
            except Exception as exc:
                yield f"data: {json.dumps({'index': i, 'status': 'error', 'message': f'Transcription failed: {exc}'})}\n\n"
            finally:
                try:
                    os.unlink(file_path)
                except OSError:
                    pass

        yield f"data: {json.dumps({'status': 'complete'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/history", response_class=HTMLResponse)
async def history():
    conn = get_db()
    if conn is None:
        return HISTORY_PAGE.replace("{{TABLE}}", '<p class="empty">Database not configured.</p>')

    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, filename, file_size, duration_s, created_at FROM transcriptions ORDER BY created_at DESC LIMIT 100"
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        return HISTORY_PAGE.replace("{{TABLE}}", '<p class="empty">No transcriptions yet.</p>')

    table_rows = ""
    for row in rows:
        tid, fname, fsize, dur, created = row
        size_mb = f"{fsize / 1024 / 1024:.1f} MB"
        duration = f"{dur:.0f}s" if dur else "—"
        date = created.strftime("%Y-%m-%d %H:%M") if created else "—"
        table_rows += (
            f"<tr>"
            f"<td>{fname}</td>"
            f"<td>{size_mb}</td>"
            f"<td>{duration}</td>"
            f"<td>{date}</td>"
            f'<td><a href="/download/{tid}">Download</a></td>'
            f"</tr>\n"
        )

    table_html = (
        "<table><thead><tr>"
        "<th>Filename</th><th>Size</th><th>Duration</th><th>Date</th><th></th>"
        "</tr></thead><tbody>\n"
        + table_rows
        + "</tbody></table>"
    )
    return HISTORY_PAGE.replace("{{TABLE}}", table_html)


@app.get("/download/{transcription_id}")
async def download(transcription_id: int):
    conn = get_db()
    if conn is None:
        raise HTTPException(status_code=503, detail="Database not configured.")

    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT filename, vtt_content FROM transcriptions WHERE id = %s",
                (transcription_id,),
            )
            row = cur.fetchone()
    finally:
        conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Transcription not found.")

    fname, vtt = row
    out_name = Path(fname).stem + ".vtt"
    return Response(
        content=vtt,
        media_type="text/vtt",
        headers={"Content-Disposition": f'attachment; filename="{out_name}"'},
    )


@app.get("/debug")
async def debug():
    status = {}

    # Env vars
    status["GOOGLE_OAUTH_TOKEN_set"] = bool(GOOGLE_OAUTH_TOKEN)
    status["AIRTABLE_TOKEN_set"] = bool(AIRTABLE_TOKEN)
    status["VIDEOS_UPLOAD_DIR"] = str(VIDEOS_UPLOAD_DIR)
    status["VIDEOS_UPLOAD_DIR_exists"] = VIDEOS_UPLOAD_DIR.is_dir()

    # ffmpeg
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True, timeout=5)
        status["ffmpeg"] = "ok"
    except Exception as exc:
        status["ffmpeg"] = f"error: {exc}"

    # Google Sheets connection
    if GOOGLE_OAUTH_TOKEN:
        try:
            import gspread
            from google.oauth2.credentials import Credentials
            from google.auth.transport.requests import Request as GRequest
            if GOOGLE_OAUTH_TOKEN.strip().startswith("{"):
                info = json.loads(GOOGLE_OAUTH_TOKEN)
                creds = Credentials.from_authorized_user_info(
                    info, scopes=["https://www.googleapis.com/auth/spreadsheets"]
                )
            else:
                creds = Credentials.from_authorized_user_file(
                    GOOGLE_OAUTH_TOKEN, scopes=["https://www.googleapis.com/auth/spreadsheets"]
                )
            if creds.expired and creds.refresh_token:
                creds.refresh(GRequest())
            gc = gspread.authorize(creds)
            ws = gc.open_by_key(UPLOAD_SHEET_ID).worksheet(UPLOAD_SHEET_TAB)
            status["sheets"] = f"ok — connected to '{ws.title}'"
        except Exception as exc:
            status["sheets"] = f"error: {exc}"
    else:
        status["sheets"] = "skipped — GOOGLE_OAUTH_TOKEN not set"

    # Master sheet reachable
    try:
        with httpx.Client(follow_redirects=True, timeout=10) as client:
            resp = client.get(_MASTER_SHEET_CSV_URL)
            resp.raise_for_status()
            row_count = len(resp.text.splitlines())
        status["master_sheet"] = f"ok — {row_count} rows"
    except Exception as exc:
        status["master_sheet"] = f"error: {exc}"

    return status
