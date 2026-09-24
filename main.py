import json
import logging
import math
import os
import re
import secrets
import tempfile
import time
import unicodedata
from pathlib import Path
from urllib.parse import unquote, urlparse

import httpx
from fastapi import FastAPI, File, Form, Request, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from pydub import AudioSegment

from openai import OpenAI
from captions import render as render_phrase_captions, tokens as caption_tokens
from starlette.concurrency import run_in_threadpool
from validation import extract_storyboard_dialogue, check_output, report_summary
import html

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

MAX_UPLOAD_SIZE = int(os.environ.get("MAX_UPLOAD_SIZE_BYTES", str(2 * 1024 * 1024 * 1024)))
CHUNK_DURATION_MS = 10 * 60 * 1000   # 10 minutes per Whisper chunk

# GPT-4o Transcribe is the higher-accuracy replacement for Whisper. Keep the
# environment override so deployments can fall back to whisper-1 if needed.
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "gpt-4o-transcribe")
CAPTION_CLEANUP_MODEL = os.environ.get("CAPTION_CLEANUP_MODEL", "gpt-4o-mini")
ENABLE_CAPTION_CLEANUP = os.environ.get("ENABLE_CAPTION_CLEANUP", "0").lower() not in {"0", "false", "no"}

CAPTION_CLEANUP_BATCH_SIZE = 20
CAPTION_CLEANUP_BATCH_CHARS = 2500
TRANSCRIPTION_MAX_ATTEMPTS = 2
MIN_VALID_CUE_DURATION_S = 0.05
MIN_VALID_TEXT_CHARS = 8

RIPTOES_OPENING_RE = re.compile(
    r"where(?:'|’)?d\s+you\s+go\s+this\s+time,?\s+riptoes\b",
    re.IGNORECASE,
)

AIRTABLE_TOKEN = os.environ.get("AIRTABLE_TOKEN", "")
AIRTABLE_BASE_ID = "appf82sOr6qFvVj6z"
AIRTABLE_TASKS_TABLE = "tblnMlOiI3q3Zj4jo"
VALIDATION_REPORTS: dict[str, tuple[float, dict]] = {}

def require_openai_api_key() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY is not set. Add your OpenAI API key to the server environment and retry.",
        )


def store_validation_report(report: dict) -> str:
    now = time.time()
    # Keep this process-local cache bounded; reports are only needed while the
    # browser renders the result table.
    for key, (created, _) in list(VALIDATION_REPORTS.items()):
        if now - created > 3600:
            VALIDATION_REPORTS.pop(key, None)
    key = secrets.token_urlsafe(18)
    VALIDATION_REPORTS[key] = (now, report)
    return key


def safe_download_filename(filename: str, fallback: str = "subtitles.vtt") -> str:
    """Build an ASCII-safe Content-Disposition filename."""
    name = unicodedata.normalize("NFKD", filename or "")
    name = name.encode("ascii", "ignore").decode("ascii")
    name = re.sub(r"[\r\n\\/\"]", "_", name).strip()
    return name or fallback


def extract_video_id(filename: str) -> int | None:
    """Parse leading Video ID from filenames like '121_What is a Community_Final.mp4'."""
    m = re.match(r'^#?(\d+)[_\s]', Path(filename).stem)
    return int(m.group(1)) if m else None


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
        return extract_storyboard_dialogue(text)
    except Exception:
        return None


def fmt_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds % 1) * 1000))
    if ms == 1000:
        s += 1
        ms = 0
    if s == 60:
        m += 1
        s = 0
    if m == 60:
        h += 1
        m = 0
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def transcribe_chunk(client: OpenAI, chunk_path: str, prompt: str | None = None):
    with open(chunk_path, "rb") as f:
        kwargs = dict(
            model=WHISPER_MODEL,
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"],
        )
        if prompt:
            kwargs["prompt"] = prompt[:900]
        return client.audio.transcriptions.create(**kwargs)


def normalize_caption_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def contains_transcription_artifact(text: str) -> bool:
    """Detect markup/formatting hallucinations that must never reach a VTT."""
    return bool(re.search(r"(?:</?(?:font|html|body|div|span|p)\b|&lt;/?(?:font|html|body|div|span|p)\b)", text or "", re.I))


def extract_json_array(text: str) -> list | None:
    text = text.strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else None
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[[\s\S]*\]", text)
    if not match:
        return None

    try:
        data = json.loads(match.group(0))
        return data if isinstance(data, list) else None
    except json.JSONDecodeError:
        return None


def cleanup_segment_batch(
    client: OpenAI, batch: list[dict], reference_text: str | None = None
) -> list[dict]:
    if not batch:
        return []

    payload = [
        {"index": i, "text": normalize_caption_text(seg.get("text", ""))}
        for i, seg in enumerate(batch)
    ]

    reference_block = (
        f"\nReference script:\n{reference_text.strip()[:1500]}\n"
        if reference_text and reference_text.strip()
        else ""
    )

    prompt = (
        "Correct these caption lines from automatic speech recognition.\n"
        "Fix capitalization and punctuation only. Preserve every spoken word.\n"
        "You do not have the audio. Never add, remove, replace, or reorder words.\n"
        "Keep the same number of items and preserve each line's meaning.\n"
        "Return JSON only as an array of strings in the same order.\n"
        f"{reference_block}\n"
        f"Caption lines:\n{json.dumps(payload, ensure_ascii=True)}"
    )

    try:
        response = client.chat.completions.create(
            model=CAPTION_CLEANUP_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You correct ASR captions conservatively and return valid JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=1200,
        )
        content = (response.choices[0].message.content or "").strip()
        corrected = extract_json_array(content)
        if not corrected or len(corrected) != len(batch):
            logger.warning("cleanup_segment_batch: invalid cleanup response, keeping original text")
            return batch
    except Exception as exc:
        logger.warning("cleanup_segment_batch failed: %s", exc)
        return batch

    cleaned = []
    for seg, text in zip(batch, corrected):
        if not isinstance(text, str) or caption_tokens(text) != caption_tokens(seg.get("text", "")):
            logger.warning("Rejected cleanup that changed spoken words")
            text = seg.get("text", "")
        cleaned.append(
            {
                **seg,
                "text": normalize_caption_text(text if isinstance(text, str) else seg.get("text", "")),
            }
        )
    return cleaned


def cleanup_caption_segments(
    client: OpenAI, segments: list[dict], reference_text: str | None = None
) -> list[dict]:
    if not ENABLE_CAPTION_CLEANUP or not segments:
        return segments

    cleaned = []
    batch = []
    batch_chars = 0

    for seg in segments:
        text = normalize_caption_text(seg.get("text", ""))
        if not text:
            cleaned.append(seg)
            continue

        batch.append(seg)
        batch_chars += len(text)
        if len(batch) >= CAPTION_CLEANUP_BATCH_SIZE or batch_chars >= CAPTION_CLEANUP_BATCH_CHARS:
            cleaned.extend(cleanup_segment_batch(client, batch, reference_text=reference_text))
            batch = []
            batch_chars = 0

    if batch:
        cleaned.extend(cleanup_segment_batch(client, batch, reference_text=reference_text))

    return cleaned


def build_caption_vtt(all_segments, all_words=None, audio_duration_s=None) -> str:
    duration = audio_duration_s if audio_duration_s is not None else max(
        (s["end"] for s in all_segments), default=0.0
    )
    vtt, report = render_phrase_captions(all_words or [], all_segments, duration, fmt_ts)
    logger.info("Caption readability: %s", report)
    return vtt


def validate_segments_for_vtt(segments: list[dict], audio_duration_s: float) -> tuple[bool, str]:
    if not segments:
        return False, "No speech segments were returned."

    text_chars = 0
    valid_ranges = 0
    prev_start = -1.0

    for seg in segments:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        text = normalize_caption_text(seg.get("text", ""))

        if not math.isfinite(start) or not math.isfinite(end):
            return False, "Non-finite segment timing."
        if start < 0 or end <= start or end > audio_duration_s + 0.05:
            return False, "Segment timing outside media bounds or empty."
        if start < prev_start - 0.001:
            return False, "Segment timings are out of order."
        prev_start = start

        if text:
            text_chars += len(text)
        if end > start:
            valid_ranges += 1

    if valid_ranges == 0:
        return False, "All returned segments had empty timing ranges."
    if text_chars < MIN_VALID_TEXT_CHARS and audio_duration_s >= 3:
        return False, "Transcript text was too short to trust."

    return True, "ok"


def parse_vtt_cues(vtt_text: str) -> list[tuple[float, float, str]]:
    cues = []
    blocks = re.split(r"\n\s*\n", vtt_text.strip())
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines or lines[0].startswith("NOTE") or lines[0] == "WEBVTT" or lines[0].startswith("Kind:") or lines[0].startswith("Language:"):
            continue

        time_index = None
        for i, line in enumerate(lines):
            if "-->" in line:
                time_index = i
                break
        if time_index is None:
            continue

        match = re.match(
            r"(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s+-->\s+(\d{2}):(\d{2}):(\d{2})\.(\d{3})",
            lines[time_index],
        )
        if not match:
            continue

        h1, m1, s1, ms1, h2, m2, s2, ms2 = map(int, match.groups())
        start = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000
        end = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000
        text = " ".join(lines[time_index + 1:]).strip()
        cues.append((start, end, text))

    return cues


def validate_vtt_output(vtt_text: str, audio_duration_s: float) -> tuple[bool, str]:
    if not vtt_text.startswith("WEBVTT"):
        return False, "VTT header missing."

    cues = parse_vtt_cues(vtt_text)
    if sum("-->" in line for line in vtt_text.splitlines()) != len(cues):
        return False, "Malformed cue timestamp."
    if not cues:
        return False, "No VTT cues were generated."

    total_text_chars = 0
    prev_end = 0.0

    for start, end, text in cues:
        if start < 0 or end > audio_duration_s + 0.001:
            return False, "Cue timing outside media bounds."
        if not text:
            return False, "Empty caption text."
        if end <= start:
            return False, "Cue timing was zero or negative."
        if end - start < MIN_VALID_CUE_DURATION_S:
            return False, "Cue timing was too short."
        if start + 0.0005 < prev_end:
            return False, "Cue timings overlap out of order."
        prev_end = end
        total_text_chars += len(normalize_caption_text(text))

    if total_text_chars < MIN_VALID_TEXT_CHARS and audio_duration_s >= 3:
        return False, "VTT text output was too short."

    return True, "ok"


def transcribe_audio_clip(client, audio, prompt=None):
    """Use lossless mono audio for short clips, compressed audio for large chunks."""
    # 10 minutes at 16 kHz mono PCM is under the transcription upload limit.
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        try:
            audio.export(tmp.name, format="wav").close()
            return transcribe_chunk(client, tmp.name, prompt=prompt)
        finally:
            os.unlink(tmp.name)


def recover_riptoes_opening(client, audio_orig):
    """Recover the spoken canned opening only when a fresh audio pass hears it."""
    pad = AudioSegment.silent(duration=2000, frame_rate=16000)
    result = transcribe_audio_clip(
        client, pad + audio_orig[:15000], prompt="Where'd you go this time Riptoes"
    )
    result_segments = getattr(result, "segments", None) or []
    confirmed = next(
        (seg for seg in result_segments if RIPTOES_OPENING_RE.search(seg.text or "")),
        None,
    )
    if confirmed is None:
        return None
    start = max(0.0, confirmed.start - 2.0)
    end = max(start + 0.05, confirmed.end - 2.0)
    words = [
        {"word": word.word, "start": max(0.0, word.start - 2.0),
         "end": max(0.0, word.end - 2.0)}
        for word in (getattr(result, "words", None) or [])
        if word.start < confirmed.end and word.end > confirmed.start
    ]
    return words, [{"start": start, "end": end,
                    "text": "Where'd you go this time, Riptoes?"}]


def audio_chunk_ranges(audio):
    """Prefer a nearby quiet boundary over cutting a word at exactly 10 minutes."""
    from pydub.silence import detect_silence
    start = 0
    while start < len(audio):
        end = min(start + CHUNK_DURATION_MS, len(audio))
        if end < len(audio):
            search_start = max(start, end - 10000)
            quiet = detect_silence(audio[search_start:end], min_silence_len=350,
                                   silence_thresh=-38, seek_step=20)
            if quiet:
                left, right = quiet[-1]
                end = search_start + (left + right) // 2
        if end <= start:
            raise ValueError("Audio chunk did not advance")
        yield start, end
        start = end


def collect_transcription_data(
    client: OpenAI, audio_orig: AudioSegment, vo_prompt: str | None = None
) -> tuple[list[dict], list[dict]]:
    audio_orig = audio_orig.set_channels(1).set_frame_rate(16000).set_sample_width(2)
    pad = AudioSegment.silent(duration=1000, frame_rate=16000)
    words, segments = [], []
    for start_ms, end_ms in audio_chunk_ranges(audio_orig):
        result = transcribe_audio_clip(client, pad + audio_orig[start_ms:end_ms], vo_prompt)
        offset = start_ms / 1000.0 - 1.0
        for word in getattr(result, "words", None) or []:
            words.append({"word": word.word, "start": max(0.0, word.start + offset),
                          "end": max(0.0, word.end + offset)})
        for seg in getattr(result, "segments", None) or []:
            segments.append({"text": seg.text, "start": max(0.0, seg.start + offset),
                             "end": max(0.0, seg.end + offset)})

    # Full-track ASR sometimes misses speech over introductory music. Re-read
    # the audio with context, without supplying or synthesizing expected dialogue.
    if segments and 3.0 <= segments[0]["start"] <= 20.0:
        boundary = segments[0]["start"]
        try:
            result = transcribe_audio_clip(client, pad + audio_orig[:int((boundary + 5) * 1000)])
            recovered = []
            for seg in getattr(result, "segments", None) or []:
                if (getattr(seg, "no_speech_prob", 0) > .5
                    or getattr(seg, "avg_logprob", 0) < -1.0
                    or getattr(seg, "compression_ratio", 0) > 2.4):
                    continue
                start, end = max(0., seg.start - 1), seg.end - 1
                if start < boundary and 0 < end <= boundary + .05:
                    recovered.append({"start": start, "end": min(end, boundary), "text": seg.text})
            if recovered:
                prefix_words = [{"word": w.word, "start": max(0., w.start - 1), "end": min(boundary, w.end - 1)}
                                for w in getattr(result, "words", None) or []
                                if any(s["start"] <= (w.start + w.end)/2 - 1 < s["end"] for s in recovered)]
                words = prefix_words + words
                segments = recovered + segments
                logger.info("Recovered %s opening speech segments from audio", len(recovered))
        except Exception as exc:
            logger.warning("Opening audio recheck failed; retaining original transcript: %s", exc)

    opening_text = " ".join(seg.get("text", "") for seg in segments if seg.get("start", 0) < 10)
    if not RIPTOES_OPENING_RE.search(opening_text):
        try:
            recovered = recover_riptoes_opening(client, audio_orig)
            if recovered:
                opening_words, opening_segments = recovered
                segments = [seg for seg in segments if seg["start"] >= 10]
                words = [word for word in words if word["start"] >= 10]
                segments = sorted(opening_segments + segments, key=lambda seg: seg["start"])
                words = sorted(opening_words + words, key=lambda word: word["start"])
                logger.info("Confirmed and restored the Riptoes opening from audio")
        except Exception as exc:
            logger.warning("Riptoes opening check failed; retaining recognized transcript: %s", exc)
    return words, segments


def transcribe_file(file_path: str, vo_prompt: str | None = None) -> str:
    client = OpenAI()
    audio_orig = AudioSegment.from_file(file_path).set_channels(1).set_frame_rate(16000)
    audio_duration_s = len(audio_orig) / 1000.0
    last_error: Exception | None = None

    for attempt in range(1, TRANSCRIPTION_MAX_ATTEMPTS + 1):
        try:
            # A storyboard prompt can bias an ASR pass toward visible text or
            # formatting artifacts. A retry without that context is safer than
            # shipping a caption containing markup.
            attempt_prompt = vo_prompt if attempt == 1 else None
            words, segments = collect_transcription_data(client, audio_orig, vo_prompt=attempt_prompt)
            ok, message = validate_segments_for_vtt(segments, audio_duration_s)
            if not ok:
                raise ValueError(message)
            raw_words = [
                {
                    "word": normalize_caption_text(word.get("word", "")),
                    "start": min(audio_duration_s, max(0.0, float(word["start"]))),
                    "end": min(audio_duration_s, max(float(word["end"]), float(word["start"]) + MIN_VALID_CUE_DURATION_S)),
                }
                for word in words
                if normalize_caption_text(word.get("word", ""))
            ]
            raw_segments = [
                {
                    "start": min(audio_duration_s, max(0.0, float(seg["start"]))),
                    "end": min(audio_duration_s, max(float(seg["end"]), float(seg["start"]) + MIN_VALID_CUE_DURATION_S)),
                    "text": normalize_caption_text(seg.get("text", "")),
                }
                for seg in segments
                if normalize_caption_text(seg.get("text", ""))
            ]

            if any(contains_transcription_artifact(seg["text"]) for seg in raw_segments):
                raise ValueError("Transcription contained markup-like artifacts; retrying without storyboard context.")

            ok, message = validate_segments_for_vtt(raw_segments, audio_duration_s)
            if not ok:
                raise ValueError(message)

            spellchecked_segments = cleanup_caption_segments(client, raw_segments, reference_text=vo_prompt)
            for label, candidate_segments in [("cleaned", spellchecked_segments), ("raw", raw_segments)]:
                vtt_text = build_caption_vtt(candidate_segments, raw_words, audio_duration_s)
                valid_vtt, vtt_message = validate_vtt_output(vtt_text, audio_duration_s)
                rendered_text = " ".join(html.unescape(t) for _, _, t in parse_vtt_cues(vtt_text))
                recognized_text = " ".join(seg["text"] for seg in raw_segments)
                if caption_tokens(rendered_text) != caption_tokens(recognized_text):
                    raise ValueError("Caption rendering changed recognized words.")
                if valid_vtt:
                    logger.info("transcribe_file: attempt %s accepted %s phrase VTT", attempt, label)
                    return vtt_text
                logger.warning("Rejected %s VTT: %s", label, vtt_message)

            raise ValueError("Generated VTT failed validation.")
        except Exception as exc:
            last_error = exc
            logger.warning("transcribe_file: attempt %s/%s failed: %s", attempt, TRANSCRIPTION_MAX_ATTEMPTS, exc)

    raise RuntimeError(
        f"Transcription failed after {TRANSCRIPTION_MAX_ATTEMPTS} attempts: {last_error}"
    )


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


def process_media_file(file_path: str, filename: str, file_size: int = 0,
                       ip: str | None = None) -> dict:
    """Generate one VTT and validate it against the video's Airtable storyboard."""
    video_id = extract_video_id(filename)
    record = fetch_airtable_record(video_id) if video_id is not None else None
    storyboard_text = fetch_storyboard_vo(record[2]) if record else None
    if video_id is None:
        logger.warning("No video ID found in filename %s; storyboard check unavailable", filename)
    elif record is None:
        logger.warning("No Airtable storyboard record found for video ID %s", video_id)
    elif not storyboard_text:
        logger.warning("Airtable record for video ID %s has no readable VO storyboard", video_id)
    vtt_text = transcribe_file(file_path, vo_prompt=storyboard_text)
    report = check_output(parse_vtt_cues(vtt_text), storyboard_text)
    validation_id = store_validation_report(report)
    return {"filename": Path(filename).stem + ".vtt",
            "vtt_text": vtt_text,
            "validation": report_summary(report),
            "validation_report": report,
            "validation_id": validation_id}


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
  .results-table { width: 100%; margin-top: 1rem; border-collapse: collapse; text-align: left; font-size: 0.8rem; }
  .results-wrap { width: 100%; overflow-x: auto; }
  .results-table th { color: #888; font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: .04em; padding: .55rem .45rem; border-bottom: 1px solid #333; }
  .results-table td { padding: .7rem .45rem; border-bottom: 1px solid #292929; vertical-align: top; }
  .results-table td:first-child { word-break: break-word; width: 170px; max-width: 170px; }
  .results-table td:nth-child(2) { width: 42%; min-width: 460px; }
  .results-table td:nth-child(3) { width: 58%; min-width: 680px; }
  .results-table a { color: #4a9eff; font-weight: 600; text-decoration: none; white-space: nowrap; }
  .validation-badge { display: inline-block; padding: .2rem .4rem; border-radius: 4px; font-size: .68rem; font-weight: 700; letter-spacing: .04em; }
  .validation-badge.matched { color: #8ee0a8; background: rgba(70, 180, 100, .16); }
  .validation-badge.review { color: #ffd27a; background: rgba(230, 160, 40, .16); }
  .validation-badge.unavailable { color: #aaa; background: rgba(150, 150, 150, .14); }
  .validation-detail { color: #aaa; margin-top: .35rem; line-height: 1.35; }
  .validation-detail details { margin-top: .35rem; }
  .validation-detail summary { color: #bbb; cursor: pointer; }
  .validation-detail .diff { margin-top: .25rem; padding: .35rem; background: #111; border-radius: 4px; }
  .line-diff { width: 100%; margin-top: .5rem; border-collapse: collapse; font-size: .75rem; }
  .line-diff th, .line-diff td { padding: .35rem; border: 1px solid #292929; vertical-align: top; text-align: left; }
  .line-diff th { color: #888; font-weight: 600; }
  .line-diff .line-number { color: #666; width: 2rem; }
  .diff-missing { color: #ff9b9b; background: rgba(210, 70, 70, .18); text-decoration: line-through; }
  .diff-extra { color: #ffd27a; background: rgba(230, 160, 40, .18); }
  .audio-cue { padding: .2rem 0; border-bottom: 1px solid #242424; }
  .audio-cue:last-child { border-bottom: 0; }
  .audio-cue small { color: #7f8b99; font-variant-numeric: tabular-nums; }
  .shot-status { white-space: nowrap; color: #b8c1cc; }
  .shot-status.matched { color: #83d6a3; }
  .shot-status.extra, .shot-status.review { color: #ffd27d; }
  .vtt-editor { display: flex; flex-direction: column; gap: .45rem; min-width: 420px; }
  .vtt-editor textarea { width: 100%; min-height: 360px; resize: vertical; box-sizing: border-box; background: #101419; color: #d8e0e8; border: 1px solid #34404d; border-radius: 6px; padding: .65rem; font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, monospace; tab-size: 2; }
  .vtt-toolbar { display: flex; align-items: center; gap: .65rem; }
  .vtt-toolbar button { width: auto; padding: .4rem .7rem; font-size: .78rem; }
  .vtt-dirty { color: #ffd27a; font-size: .72rem; }
  .validation-detail { max-height: 500px; overflow: auto; padding-right: .25rem; }
  .line-diff { min-width: 620px; }

  .batch-log { margin-top: 1rem; font-size: 0.85rem; max-height: 300px; overflow-y: auto; }
  .batch-section { margin-bottom: 1.25rem; }
  .batch-section h3 { font-size: 0.95rem; margin-bottom: 0.5rem; color: #ccc; }
  .batch-hint { color: #777; font-size: 0.8rem; margin-bottom: 0.75rem; }
  .batch-item { padding: 0.5rem 0; border-bottom: 1px solid #222; display: flex; justify-content: space-between; align-items: center; }
  .batch-item .name { color: #ccc; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; margin-right: 0.5rem; }
  .batch-item .state { font-size: 0.8rem; white-space: normal; max-width: 65%; }
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
  <p class="sub">Generate cleaner subtitle files (.vtt) with steadier cue timing</p>

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
        <input type="file" id="file-input" multiple accept="video/*,audio/*,.mp3,.mp4,.m4a,.wav,.webm,.ogg,.flac,.mpeg,.mpga">
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
    <div class="batch-section">
      <h3>Batch Upload</h3>
      <p class="batch-hint">Select several video or audio files and process them one after another.</p>
      <form id="batch-upload-form">
        <label class="file-label" id="batch-drop-label" for="batch-file-input">
          <span id="batch-label-text">Click to select or drag multiple files here</span>
          <div class="file-name" id="batch-file-name"></div>
          <input type="file" id="batch-file-input" multiple accept="video/*,audio/*,.mp3,.mp4,.m4a,.wav,.webm,.ogg,.flac,.mpeg,.mpga">
        </label>
        <button type="submit" id="batch-upload-btn" disabled>Process Uploaded Files</button>
      </form>
    </div>

    <div class="batch-section">
      <h3>Batch URLs</h3>
      <p class="batch-hint">Paste direct media URLs, one per line.</p>
      <form id="batch-form">
        <textarea id="batch-input" placeholder="Paste URLs, one per line"></textarea>
        <button type="submit" id="batch-btn">Process URL List</button>
      </form>
    </div>
    <div class="batch-log" id="batch-log"></div>
  </div>


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

  function setFiles(files) {
    const dt = new DataTransfer();
    Array.from(files).forEach(file => dt.items.add(file));
    fileInput.files = dt.files;
    const totalMb = Array.from(files).reduce((sum, file) => sum + file.size, 0) / 1024 / 1024;
    fileNameEl.textContent = files.length + ' file' + (files.length === 1 ? '' : 's') + ' (' + totalMb.toFixed(1) + ' MB)';
    label.classList.add('has-file');
    fileBtn.disabled = files.length === 0;
  }

  function parseValidation(raw) {
    if (!raw) return {};
    try {
      return JSON.parse(raw);
    } catch (_) { return {storyboard: {status: 'review', message: raw}}; }
  }

  function createResultsTable(container) {
    container.innerHTML = '';
    container.classList.add('results-wrap');
    const table = document.createElement('table');
    table.className = 'results-table';
    table.innerHTML = '<thead><tr><th>File</th><th>VTT</th><th>Validation</th></tr></thead>';
    const body = document.createElement('tbody');
    table.appendChild(body);
    container.appendChild(table);
    return body;
  }

  function fillValidationCell(validationCell, rawReport, error) {
    const report = parseValidation(rawReport);
    const storyboard = report.storyboard || {};
    if (error) {
      validationCell.textContent = error;
    } else {
      const status = storyboard.status || 'unavailable';
      const badge = document.createElement('span');
      badge.className = 'validation-badge ' + status;
      badge.textContent = status === 'matched' ? 'MATCHED' : status === 'review' ? 'REVIEW' : 'NOT CHECKED';
      validationCell.appendChild(badge);
      const detail = document.createElement('div');
      detail.className = 'validation-detail';
      const readabilityIssues = (report.short_cues || 0) + (report.fast_cues || 0) + (report.long_lines || 0);
      detail.textContent = (storyboard.message || 'Validation completed.') +
        (readabilityIssues ? ' Readability needs review.' : ' Readability passed.');
      validationCell.appendChild(detail);
      if (storyboard.lines && storyboard.lines.length) {
        const details = document.createElement('details');
        details.open = storyboard.status === 'review';
        const summary = document.createElement('summary'); summary.textContent = 'Shot-by-shot audio comparison'; details.appendChild(summary);
        const table = document.createElement('table'); table.className = 'line-diff';
        table.innerHTML = '<thead><tr><th>Shot</th><th>Storyboard VO</th><th>Audio (dialogue)</th><th>Status</th></tr></thead>';
        const body = document.createElement('tbody'); table.appendChild(body);
        const addParts = (cell, parts) => parts.forEach(part => {
          const span = document.createElement('span');
          if (part.kind !== 'same') span.className = part.kind === 'missing' ? 'diff-missing' : 'diff-extra';
          span.textContent = part.text + ' ';
          cell.appendChild(span);
        });
        storyboard.lines.forEach(line => {
          const row = document.createElement('tr');
          const number = document.createElement('td'); number.className = 'line-number'; number.textContent = line.line;
          const storyboardCell = document.createElement('td'); addParts(storyboardCell, line.storyboard_parts || []);
          const transcriptCell = document.createElement('td');
          const cues = line.audio_cues || [];
          if (cues.length) {
            cues.forEach(cue => {
              const cueBlock = document.createElement('div'); cueBlock.className = 'audio-cue';
              const time = document.createElement('small');
              time.textContent = cue.start != null ? ('[' + Number(cue.start).toFixed(2) + '–' + Number(cue.end).toFixed(2) + '] ') : '';
              cueBlock.appendChild(time);
              addParts(cueBlock, cue.transcript_parts || [{text: cue.text || '', kind: 'extra'}]);
              transcriptCell.appendChild(cueBlock);
            });
          } else { transcriptCell.textContent = 'No audio cue aligned'; }
          const status = document.createElement('td'); status.textContent = line.status === 'matched' ? 'Matched' : line.status === 'extra' ? 'Audio not in storyboard' : 'Review';
          status.className = 'shot-status ' + line.status;
          row.append(number, storyboardCell, transcriptCell, status); body.appendChild(row);
        });
        details.appendChild(table);
        validationCell.appendChild(details);
      }
    }
  }

  function addResultRow(body, filename, blobUrl, rawReport, error, vttText) {
    const row = document.createElement('tr');
    const fileCell = document.createElement('td');
    fileCell.textContent = filename;
    const linkCell = document.createElement('td');
    if (blobUrl || vttText) {
      const editor = document.createElement('div'); editor.className = 'vtt-editor';
      const textarea = document.createElement('textarea');
      textarea.value = vttText || '';
      textarea.setAttribute('aria-label', 'Editable VTT for ' + filename);
      const link = document.createElement('a');
      link.download = filename.replace(/[^.]+$/, '') + 'vtt'; link.textContent = 'Download VTT';
      let currentUrl = blobUrl;
      const toolbar = document.createElement('div'); toolbar.className = 'vtt-toolbar';
      const dirty = document.createElement('span'); dirty.className = 'vtt-dirty'; dirty.textContent = 'Click in the editor to make corrections.';
      toolbar.append(link, dirty);
      textarea.addEventListener('input', () => {
        if (currentUrl) URL.revokeObjectURL(currentUrl);
        currentUrl = URL.createObjectURL(new Blob([textarea.value], {type: 'text/vtt;charset=utf-8'}));
        link.href = currentUrl;
        dirty.textContent = 'Edited — download includes your changes.';
      });
      editor.append(textarea, toolbar); linkCell.appendChild(editor);
    } else { linkCell.textContent = 'Failed'; }
    const validationCell = document.createElement('td');
    fillValidationCell(validationCell, rawReport, error);
    row.append(fileCell, linkCell, validationCell);
    body.appendChild(row);
    return validationCell;
  }

  async function loadValidationCell(cell, validationId) {
    if (!validationId) { fillValidationCell(cell, ''); return; }
    cell.textContent = 'Loading validation…';
    try {
      const res = await fetch('/validation/' + encodeURIComponent(validationId));
      if (!res.ok) throw new Error('Validation report unavailable');
      fillValidationCell(cell, JSON.stringify(await res.json()));
    } catch (err) {
      cell.textContent = err.message;
    }
  }

  async function responseError(res, fallback) {
    const text = await res.text();
    try {
      const data = JSON.parse(text);
      return data.detail || fallback;
    } catch (_) {
      return text || fallback;
    }
  }

  fileInput.addEventListener('change', () => setFiles(fileInput.files));

  label.addEventListener('dragover', (e) => { e.preventDefault(); e.stopPropagation(); label.classList.add('dragover'); });
  label.addEventListener('dragleave', (e) => { e.preventDefault(); e.stopPropagation(); label.classList.remove('dragover'); });
  label.addEventListener('drop', (e) => {
    e.preventDefault(); e.stopPropagation(); label.classList.remove('dragover');
    if (e.dataTransfer.files.length) setFiles(e.dataTransfer.files);
  });

  fileForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const files = Array.from(fileInput.files || []);
    if (!files.length) return;
    fileBtn.disabled = true;
    fileStatus.className = 'status';
    fileStatus.textContent = 'Transcribing ' + files.length + ' file' + (files.length === 1 ? '' : 's') + '…';
    try {
      fileStatus.className = 'status success';
      const resultBody = createResultsTable(fileStatus);
      for (const file of files) {
        const fd = new FormData();
        fd.append('file', file);
        const res = await fetch('/transcribe', { method: 'POST', body: fd });
        if (!res.ok) throw new Error(file.name + ': ' + await responseError(res, 'Transcription failed'));
        const vttText = await res.text();
        const url = URL.createObjectURL(new Blob([vttText], {type: 'text/vtt;charset=utf-8'}));
        const cell = addResultRow(resultBody, file.name, url, '', null, vttText);
        await loadValidationCell(cell, res.headers.get('X-Validation-ID'));
      }
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
      if (!res.ok) throw new Error(await responseError(res, 'Transcription failed'));
      const vttText = await res.text();
      const disp = res.headers.get('Content-Disposition') || '';
      const match = disp.match(/filename="(.+?)"/);
      const name = match ? match[1] : 'subtitles.vtt';
      const blobUrl = URL.createObjectURL(new Blob([vttText], {type: 'text/vtt;charset=utf-8'}));
      urlStatus.className = 'status success';
      const resultBody = createResultsTable(urlStatus);
      const cell = addResultRow(resultBody, name, blobUrl, '', null, vttText);
      await loadValidationCell(cell, res.headers.get('X-Validation-ID'));
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
  const batchUploadForm = document.getElementById('batch-upload-form');
  const batchFileInput = document.getElementById('batch-file-input');
  const batchFileNameEl = document.getElementById('batch-file-name');
  const batchDropLabel = document.getElementById('batch-drop-label');
  const batchUploadBtn = document.getElementById('batch-upload-btn');

  function formatFilesSummary(files) {
    const totalMb = Array.from(files).reduce((sum, f) => sum + f.size, 0) / 1024 / 1024;
    return files.length + ' files (' + totalMb.toFixed(1) + ' MB)';
  }

  function renderBatchItems(items) {
    batchLog.innerHTML = '';
    const table = document.createElement('table');
    table.className = 'results-table';
    table.innerHTML = '<thead><tr><th>File</th><th>VTT</th><th>Validation</th></tr></thead>';
    const body = document.createElement('tbody');
    table.appendChild(body);
    batchLog.appendChild(table);
    items.forEach((item, i) => {
      const row = document.createElement('tr');
      const name = document.createElement('td'); name.textContent = item.name; name.title = item.title;
      const download = document.createElement('td');
      const state = document.createElement('span'); state.className = 'state queued'; state.id = 'bs-' + i; state.textContent = 'queued';
      download.appendChild(state);
      const validation = document.createElement('td'); validation.id = 'bv-' + i;
      row.append(name, download, validation); body.appendChild(row);
    });
  }

  function updateBatchState(index, status, html) {
    const el = document.getElementById('bs-' + index);
    if (!el) return;
    el.className = 'state ' + status;
    if (html !== undefined) el.innerHTML = html;
  }

  function setBatchEditor(index, text, filename) {
    const state = document.getElementById('bs-' + index);
    if (!state) return;
    const cell = state.closest('td'); cell.innerHTML = '';
    const editor = document.createElement('div'); editor.className = 'vtt-editor';
    const textarea = document.createElement('textarea'); textarea.value = text;
    textarea.setAttribute('aria-label', 'Editable VTT for ' + filename);
    const link = document.createElement('a'); link.download = filename; link.textContent = 'Download VTT';
    let url = URL.createObjectURL(new Blob([text], {type: 'text/vtt;charset=utf-8'})); link.href = url;
    const dirty = document.createElement('span'); dirty.className = 'vtt-dirty'; dirty.textContent = 'Edit, then download.';
    textarea.addEventListener('input', () => {
      if (url) URL.revokeObjectURL(url);
      url = URL.createObjectURL(new Blob([textarea.value], {type: 'text/vtt;charset=utf-8'}));
      link.href = url; dirty.textContent = 'Edited — download includes your changes.';
    });
    const toolbar = document.createElement('div'); toolbar.className = 'vtt-toolbar'; toolbar.append(link, dirty);
    editor.append(textarea, toolbar); cell.appendChild(editor);
  }

  async function setBatchValidation(index, validationId) {
    const cell = document.getElementById('bv-' + index);
    if (!cell) return;
    await loadValidationCell(cell, validationId);
  }

  function setBatchFiles(files) {
    batchFileInput.files = files;
    batchFileNameEl.textContent = files.length ? formatFilesSummary(files) : '';
    batchDropLabel.classList.toggle('has-file', files.length > 0);
    batchUploadBtn.disabled = files.length === 0;
  }

  batchFileInput.addEventListener('change', () => {
    setBatchFiles(batchFileInput.files);
  });

  batchDropLabel.addEventListener('dragover', (e) => { e.preventDefault(); e.stopPropagation(); batchDropLabel.classList.add('dragover'); });
  batchDropLabel.addEventListener('dragleave', (e) => { e.preventDefault(); e.stopPropagation(); batchDropLabel.classList.remove('dragover'); });
  batchDropLabel.addEventListener('drop', (e) => {
    e.preventDefault(); e.stopPropagation(); batchDropLabel.classList.remove('dragover');
    if (!e.dataTransfer.files.length) return;
    const dt = new DataTransfer();
    Array.from(e.dataTransfer.files).forEach(file => dt.items.add(file));
    setBatchFiles(dt.files);
  });

  batchUploadForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const files = Array.from(batchFileInput.files || []);
    if (!files.length) return;

    batchUploadBtn.disabled = true;
    batchBtn.disabled = true;

    const items = files.map(file => ({ name: file.name, title: file.name }));
    renderBatchItems(items);

    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      updateBatchState(i, 'transcribing', 'uploading…');
      try {
        const fd = new FormData();
        fd.append('file', file);
        const res = await fetch('/transcribe', { method: 'POST', body: fd });
        if (!res.ok) {
          let message = 'Transcription failed';
          try {
            message = await responseError(res, message);
          } catch (_) {}
          throw new Error(message);
        }
        const vttText = await res.text();
        const name = file.name.replace(/\\.[^.]+$/, '') + '.vtt';
        setBatchEditor(i, vttText, name);
        await setBatchValidation(i, res.headers.get('X-Validation-ID'));
      } catch (err) {
        updateBatchState(i, 'error', (err && err.message) ? err.message : 'error');
      }
    }

    batchUploadBtn.disabled = false;
    batchBtn.disabled = false;
  });

  batchForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const urls = batchInput.value.trim().split('\\n').map(u => u.trim()).filter(Boolean);
    if (!urls.length) return;
    batchBtn.disabled = true;
    batchUploadBtn.disabled = true;

    // Build item rows
    const items = {};
    renderBatchItems(urls.map((url, i) => ({
      name: url.split('/').pop().split('?')[0] || ('file-' + (i+1)),
      title: url
    })));
    urls.forEach((url, i) => {
      items[i] = { url };
    });

    const es = new EventSource('/batch?urls=' + encodeURIComponent(JSON.stringify(urls)));
    es.onmessage = (evt) => {
      const d = JSON.parse(evt.data);
      if (d.status === 'downloading') {
        updateBatchState(d.index, 'downloading', 'downloading…');
      } else if (d.status === 'transcribing') {
        updateBatchState(d.index, 'transcribing', 'transcribing…');
      } else if (d.status === 'done') {
        setBatchEditor(d.index, d.vtt_text, 'subtitles-' + (d.index + 1) + '.vtt');
        setBatchValidation(d.index, d.validation_id);
      } else if (d.status === 'error') {
        updateBatchState(d.index, 'error', d.message || 'error');
      } else if (d.status === 'complete') {
        es.close();
        batchBtn.disabled = false;
        batchUploadBtn.disabled = batchFileInput.files.length === 0;
      }
    };
    es.onerror = () => { es.close(); batchBtn.disabled = false; batchUploadBtn.disabled = batchFileInput.files.length === 0; };
  });
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


@app.post("/transcribe")
async def transcribe(request: Request, file: UploadFile = File(...)):
    require_openai_api_key()
    original_name = file.filename or "unknown"
    suffix = Path(file.filename).suffix if file.filename else ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    file_size = 0
    try:
        while chunk := await file.read(1024 * 1024):
            file_size += len(chunk)
            if file_size > MAX_UPLOAD_SIZE:
                tmp.close()
                raise HTTPException(
                    status_code=413,
                    detail=f"File exceeds the {MAX_UPLOAD_SIZE / 1024 / 1024 / 1024:.0f} GB upload limit.",
                )
            tmp.write(chunk)
        tmp.close()
        ip = request.client.host if request.client else None
        result = await run_in_threadpool(process_media_file, tmp.name, original_name, file_size, ip)

        out_name = safe_download_filename(Path(original_name).stem + ".vtt")
        resp_headers = {"Content-Disposition": f'attachment; filename="{out_name}"'}
        resp_headers["X-Validation-Summary"] = result.get("validation", "").encode("ascii", "ignore").decode("ascii")
        resp_headers["X-Validation-ID"] = result.get("validation_id", "")
        return Response(content=result["vtt_text"], media_type="text/vtt", headers=resp_headers)
    finally:
        os.unlink(tmp.name)


@app.post("/transcribe-url")
async def transcribe_url(request: Request, url: str = Form(...)):
    require_openai_api_key()
    try:
        file_path, filename, file_size = await run_in_threadpool(download_url_to_temp, url)
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=400, detail=f"Download failed: HTTP {exc.response.status_code}")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Download failed: {exc}")

    try:
        ip = request.client.host if request.client else None
        result = await run_in_threadpool(process_media_file, file_path, filename, file_size, ip)

        out_name = safe_download_filename(Path(filename).stem + ".vtt")
        resp_headers = {"Content-Disposition": f'attachment; filename="{out_name}"'}
        resp_headers["X-Validation-Summary"] = result.get("validation", "").encode("ascii", "ignore").decode("ascii")
        resp_headers["X-Validation-ID"] = result.get("validation_id", "")
        return Response(content=result["vtt_text"], media_type="text/vtt", headers=resp_headers)
    finally:
        os.unlink(file_path)


@app.get("/validation/{validation_id}")
async def validation_report(validation_id: str):
    entry = VALIDATION_REPORTS.get(validation_id)
    if not entry or time.time() - entry[0] > 3600:
        VALIDATION_REPORTS.pop(validation_id, None)
        raise HTTPException(status_code=404, detail="Validation report expired or was not found.")
    return entry[1]


@app.get("/batch")
async def batch(request: Request, urls: str):
    """SSE endpoint: processes a list of URLs and streams progress events."""
    require_openai_api_key()
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
                result = process_media_file(file_path, filename, file_size, ip)
                result_for_ui = {key: value for key, value in result.items() if key != 'validation_report'}
                yield f"data: {json.dumps({'index': i, 'status': 'done', **result_for_ui})}\n\n"
            except Exception as exc:
                yield f"data: {json.dumps({'index': i, 'status': 'error', 'message': f'Transcription failed: {exc}'})}\n\n"
            finally:
                try:
                    os.unlink(file_path)
                except OSError:
                    pass

        yield f"data: {json.dumps({'status': 'complete'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
