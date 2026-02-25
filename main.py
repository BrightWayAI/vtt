import os
import re
import tempfile
from pathlib import Path

import psycopg2
from fastapi import FastAPI, File, Request, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydub import AudioSegment

from openai import OpenAI

app = FastAPI()

MAX_FILE_SIZE = 500 * 1024 * 1024  # 500 MB
CHUNK_DURATION_MS = 10 * 60 * 1000  # 10 minutes per chunk

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
    """Extract duration from the last timestamp in VTT content."""
    matches = re.findall(r"(\d{2}):(\d{2}):(\d{2})\.(\d{3})", vtt_text)
    if not matches:
        return None
    h, m, s, ms = matches[-1]
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def fmt_ts(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds % 1) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def transcribe_chunk(client: OpenAI, chunk_path: str):
    """Transcribe a single audio chunk via Whisper and return the result."""
    with open(chunk_path, "rb") as f:
        return client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"],
        )


def build_word_highlight_vtt(all_words, all_segments) -> str:
    """Build a WebVTT file with per-word highlight cues.

    Each word gets its own cue showing the full segment text with the
    active word wrapped in <v>...</v> tags.
    """
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


def transcribe_file(file_path: str) -> str:
    """Transcribe an audio/video file, splitting into chunks if needed.

    Returns the complete WebVTT string.
    """
    client = OpenAI()
    audio = AudioSegment.from_file(file_path)
    duration_ms = len(audio)

    # If short enough, send directly without splitting
    if duration_ms <= CHUNK_DURATION_MS:
        # Export as mp3 to keep under 25MB for Whisper
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            audio.export(tmp.name, format="mp3", bitrate="64k")
            result = transcribe_chunk(client, tmp.name)
            os.unlink(tmp.name)

        words = [{"word": w.word, "start": w.start, "end": w.end}
                 for w in (getattr(result, "words", None) or [])]
        segments = [{"start": s.start, "end": s.end, "text": s.text}
                    for s in (getattr(result, "segments", None) or [])]
        return build_word_highlight_vtt(words, segments)

    # Split into chunks
    all_words = []
    all_segments = []

    for chunk_start_ms in range(0, duration_ms, CHUNK_DURATION_MS):
        chunk_end_ms = min(chunk_start_ms + CHUNK_DURATION_MS, duration_ms)
        chunk = audio[chunk_start_ms:chunk_end_ms]
        offset_s = chunk_start_ms / 1000.0

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            chunk.export(tmp.name, format="mp3", bitrate="64k")
            result = transcribe_chunk(client, tmp.name)
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

    return build_word_highlight_vtt(all_words, all_segments)


def save_transcription(filename: str, file_size: int, vtt_content: str, ip: str | None):
    conn = get_db()
    if conn is None:
        return
    try:
        duration = parse_vtt_duration(vtt_content)
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO transcriptions (filename, file_size, duration_s, vtt_content, ip_address)
                   VALUES (%s, %s, %s, %s, %s)""",
                (filename, file_size, duration, vtt_content, ip),
            )
        conn.commit()
    finally:
        conn.close()


@app.on_event("startup")
def startup():
    init_db()


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
  .container { background: #1a1a1a; border-radius: 12px; padding: 2.5rem; max-width: 480px; width: 90%; box-shadow: 0 4px 24px rgba(0,0,0,0.4); }
  h1 { font-size: 1.5rem; margin-bottom: 0.5rem; }
  p.sub { color: #888; margin-bottom: 1.5rem; font-size: 0.9rem; }
  label.file-label { display: block; border: 2px dashed #333; border-radius: 8px; padding: 2rem; text-align: center; cursor: pointer; transition: border-color 0.2s; margin-bottom: 1rem; }
  label.file-label:hover { border-color: #555; }
  label.file-label.has-file { border-color: #4a9eff; }
  label.file-label.dragover { border-color: #4a9eff; background: rgba(74,158,255,0.05); }
  input[type="file"] { display: none; }
  .file-name { font-size: 0.85rem; color: #4a9eff; margin-top: 0.5rem; word-break: break-all; }
  button { width: 100%; padding: 0.75rem; border: none; border-radius: 8px; background: #4a9eff; color: #fff; font-size: 1rem; cursor: pointer; transition: background 0.2s; }
  button:hover { background: #3a8eef; }
  button:disabled { background: #333; color: #666; cursor: not-allowed; }
  .status { margin-top: 1rem; font-size: 0.9rem; text-align: center; }
  .status.error { color: #ff6b6b; }
  .status.success a { color: #4a9eff; text-decoration: none; font-weight: 600; }
  .nav { margin-top: 1.5rem; text-align: center; }
  .nav a { color: #888; font-size: 0.85rem; text-decoration: none; }
  .nav a:hover { color: #4a9eff; }
</style>
</head>
<body>
<div class="container">
  <h1>VTT Generator</h1>
  <p class="sub">Upload a video or audio file to generate subtitles (.vtt)</p>
  <form id="form">
    <label class="file-label" id="drop-label" for="file-input">
      <span id="label-text">Click to select or drag a file here</span>
      <div class="file-name" id="file-name"></div>
      <input type="file" id="file-input" accept="video/*,audio/*,.mp3,.mp4,.m4a,.wav,.webm,.ogg,.flac,.mpeg,.mpga">
    </label>
    <button type="submit" id="submit-btn" disabled>Generate VTT</button>
  </form>
  <div class="status" id="status"></div>
  <div class="nav"><a href="/history">View transcription history</a></div>
</div>
<script>
  const form = document.getElementById('form');
  const fileInput = document.getElementById('file-input');
  const fileNameEl = document.getElementById('file-name');
  const label = document.getElementById('drop-label');
  const btn = document.getElementById('submit-btn');
  const status = document.getElementById('status');

  function setFile(file) {
    const dt = new DataTransfer();
    dt.items.add(file);
    fileInput.files = dt.files;
    fileNameEl.textContent = file.name + ' (' + (file.size / 1024 / 1024).toFixed(1) + ' MB)';
    label.classList.add('has-file');
    btn.disabled = false;
  }

  fileInput.addEventListener('change', () => {
    if (fileInput.files.length) setFile(fileInput.files[0]);
  });

  // Drag and drop
  label.addEventListener('dragover', (e) => {
    e.preventDefault();
    e.stopPropagation();
    label.classList.add('dragover');
  });
  label.addEventListener('dragleave', (e) => {
    e.preventDefault();
    e.stopPropagation();
    label.classList.remove('dragover');
  });
  label.addEventListener('drop', (e) => {
    e.preventDefault();
    e.stopPropagation();
    label.classList.remove('dragover');
    if (e.dataTransfer.files.length) setFile(e.dataTransfer.files[0]);
  });

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const file = fileInput.files[0];
    if (!file) return;
    if (file.size > 500 * 1024 * 1024) {
      status.className = 'status error';
      status.textContent = 'File exceeds 500 MB limit.';
      return;
    }
    btn.disabled = true;
    status.className = 'status';
    status.textContent = 'Transcribing… this may take a few minutes for large files.';
    try {
      const fd = new FormData();
      fd.append('file', file);
      const res = await fetch('/transcribe', { method: 'POST', body: fd });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Transcription failed');
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const name = file.name.replace(/\\.[^.]+$/, '') + '.vtt';
      status.className = 'status success';
      status.innerHTML = '<a href="' + url + '" download="' + name + '">Download ' + name + '</a>';
    } catch (err) {
      status.className = 'status error';
      status.textContent = err.message;
    } finally {
      btn.disabled = false;
    }
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
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File exceeds 500 MB limit.")

    suffix = Path(file.filename).suffix if file.filename else ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(contents)
        tmp.close()

        vtt_text = transcribe_file(tmp.name)

        # Save to database
        ip = request.client.host if request.client else None
        original_name = file.filename or "unknown"
        save_transcription(original_name, len(contents), vtt_text, ip)

        out_name = Path(file.filename).stem + ".vtt" if file.filename else "subtitles.vtt"
        return Response(
            content=vtt_text,
            media_type="text/vtt",
            headers={"Content-Disposition": f'attachment; filename="{out_name}"'},
        )
    finally:
        os.unlink(tmp.name)


@app.get("/history", response_class=HTMLResponse)
async def history():
    conn = get_db()
    if conn is None:
        html = HISTORY_PAGE.replace("{{TABLE}}", '<p class="empty">Database not configured.</p>')
        return html

    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, filename, file_size, duration_s, created_at FROM transcriptions ORDER BY created_at DESC LIMIT 100"
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        html = HISTORY_PAGE.replace("{{TABLE}}", '<p class="empty">No transcriptions yet.</p>')
        return html

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
