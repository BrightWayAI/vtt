import json
import os
import re
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlparse

import httpx
import psycopg2
from fastapi import FastAPI, File, Form, Request, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from pydub import AudioSegment

from openai import OpenAI

app = FastAPI()

MAX_UPLOAD_SIZE = 500 * 1024 * 1024  # 500 MB for direct uploads
CHUNK_DURATION_MS = 10 * 60 * 1000   # 10 minutes per Whisper chunk

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
# Transcription helpers
# ---------------------------------------------------------------------------

def fmt_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds % 1) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def transcribe_chunk(client: OpenAI, chunk_path: str):
    with open(chunk_path, "rb") as f:
        return client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"],
        )


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


def transcribe_file(file_path: str) -> str:
    client = OpenAI()
    audio = AudioSegment.from_file(file_path)
    duration_ms = len(audio)

    if duration_ms <= CHUNK_DURATION_MS:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            audio.export(tmp.name, format="mp3", bitrate="64k")
            result = transcribe_chunk(client, tmp.name)
            os.unlink(tmp.name)

        words = [{"word": w.word, "start": w.start, "end": w.end}
                 for w in (getattr(result, "words", None) or [])]
        segments = [{"start": s.start, "end": s.end, "text": s.text}
                    for s in (getattr(result, "segments", None) or [])]
        return build_word_highlight_vtt(words, segments)

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


def download_url_to_temp(url: str) -> tuple[str, str, int]:
    """Download a URL to a temp file. Returns (path, filename, size)."""
    parsed = urlparse(url)
    filename = Path(unquote(parsed.path)).name or "download.mp4"
    suffix = Path(filename).suffix or ".mp4"

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    size = 0
    with httpx.stream("GET", url, follow_redirects=True, timeout=300) as r:
        r.raise_for_status()
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
      const res = await fetch('/transcribe-url', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: 'url=' + encodeURIComponent(url),
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

    suffix = Path(file.filename).suffix if file.filename else ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(contents)
        tmp.close()

        vtt_text = transcribe_file(tmp.name)

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


@app.post("/transcribe-url")
async def transcribe_url(request: Request, url: str = Form(...)):
    try:
        file_path, filename, file_size = download_url_to_temp(url)
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=400, detail=f"Download failed: HTTP {exc.response.status_code}")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Download failed: {exc}")

    try:
        vtt_text = transcribe_file(file_path)

        ip = request.client.host if request.client else None
        save_transcription(filename, file_size, vtt_text, ip)

        out_name = Path(filename).stem + ".vtt"
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
                vtt_text = transcribe_file(file_path)
                tid = save_transcription(filename, file_size, vtt_text, ip)
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
        duration = f"{dur:.0f}s" if dur else "\u2014"
        date = created.strftime("%Y-%m-%d %H:%M") if created else "\u2014"
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
