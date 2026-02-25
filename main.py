import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, Response
from openai import OpenAI

app = FastAPI()

MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB (Whisper API limit)

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
  input[type="file"] { display: none; }
  .file-name { font-size: 0.85rem; color: #4a9eff; margin-top: 0.5rem; word-break: break-all; }
  button { width: 100%; padding: 0.75rem; border: none; border-radius: 8px; background: #4a9eff; color: #fff; font-size: 1rem; cursor: pointer; transition: background 0.2s; }
  button:hover { background: #3a8eef; }
  button:disabled { background: #333; color: #666; cursor: not-allowed; }
  .status { margin-top: 1rem; font-size: 0.9rem; text-align: center; }
  .status.error { color: #ff6b6b; }
  .status.success a { color: #4a9eff; text-decoration: none; font-weight: 600; }
</style>
</head>
<body>
<div class="container">
  <h1>VTT Generator</h1>
  <p class="sub">Upload a video or audio file to generate subtitles (.vtt)</p>
  <form id="form">
    <label class="file-label" id="drop-label">
      <span id="label-text">Click to select or drag a file here</span>
      <div class="file-name" id="file-name"></div>
      <input type="file" id="file-input" accept="video/*,audio/*,.mp3,.mp4,.m4a,.wav,.webm,.ogg,.flac,.mpeg,.mpga">
    </label>
    <button type="submit" id="submit-btn" disabled>Generate VTT</button>
  </form>
  <div class="status" id="status"></div>
</div>
<script>
  const form = document.getElementById('form');
  const fileInput = document.getElementById('file-input');
  const fileName = document.getElementById('file-name');
  const label = document.getElementById('drop-label');
  const btn = document.getElementById('submit-btn');
  const status = document.getElementById('status');

  fileInput.addEventListener('change', () => {
    if (fileInput.files.length) {
      const f = fileInput.files[0];
      fileName.textContent = f.name + ' (' + (f.size / 1024 / 1024).toFixed(1) + ' MB)';
      label.classList.add('has-file');
      btn.disabled = false;
    }
  });

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const file = fileInput.files[0];
    if (!file) return;
    if (file.size > 25 * 1024 * 1024) {
      status.className = 'status error';
      status.textContent = 'File exceeds 25 MB limit.';
      return;
    }
    btn.disabled = true;
    status.className = 'status';
    status.textContent = 'Transcribing… this may take a minute.';
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


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File exceeds 25 MB limit.")

    suffix = Path(file.filename).suffix if file.filename else ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(contents)
        tmp.close()

        client = OpenAI()
        with open(tmp.name, "rb") as audio_file:
            vtt_text = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="vtt",
            )

        out_name = Path(file.filename).stem + ".vtt" if file.filename else "subtitles.vtt"
        return Response(
            content=vtt_text,
            media_type="text/vtt",
            headers={"Content-Disposition": f'attachment; filename="{out_name}"'},
        )
    finally:
        os.unlink(tmp.name)
