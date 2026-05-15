import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import httpx
from openai import OpenAI


AIRTABLE_BASE_ID = "appf82sOr6qFvVj6z"
AIRTABLE_TASKS_TABLE = "tblnMlOiI3q3Zj4jo"
VTT_DIR = Path.home() / "Desktop" / "Videos for Upload"
SKIP_VIDEO_IDS = {30}
MAX_LINE_CHARS = 45
MIN_CHUNK_CHARS = 18
TARGET_CHUNK_CHARS = 38
MIN_SUBTITLE_DURATION = 0.8
TAIL_TRANSCRIBE_SECONDS = 35

SPEAKER_RE = re.compile(r"(NARRATOR|ALMA|MATEO|DIALOGUE)\s*:\s*", re.IGNORECASE)
DOC_ID_RE = re.compile(r"/document/d/([a-zA-Z0-9_-]+)")


@dataclass
class Cue:
    start: str
    end: str
    text: str

    @property
    def duration(self) -> float:
        return ts_to_seconds(self.end) - ts_to_seconds(self.start)


def ts_to_seconds(ts: str) -> float:
    h, m, rest = ts.split(":")
    s, ms = rest.split(".")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def seconds_to_ts(seconds: float) -> str:
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


def normalize_storyboard_text(text: str) -> str:
    text = text.replace("\ufeff", "")
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2014", "-").replace("\u2013", "-")
    text = re.sub(r"\[SILENCE[^\]]*\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text.strip().strip('"')


def extract_video_id(filename: str) -> int | None:
    match = re.match(r"^(\d+)[_\s]", Path(filename).stem)
    return int(match.group(1)) if match else None


def fetch_storyboard_link(client: httpx.Client, video_id: int) -> tuple[str, str]:
    resp = client.get(
        f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TASKS_TABLE}",
        params={
            "filterByFormula": f"{{Video ID #}}={video_id}",
            "fields[]": ["Video Topic", "Storyboard Link"],
            "maxRecords": "1",
        },
        headers={"Authorization": f"Bearer {os.environ['AIRTABLE_TOKEN']}"},
    )
    resp.raise_for_status()
    records = resp.json().get("records", [])
    if not records:
        raise RuntimeError(f"No Airtable record for video {video_id}")
    fields = records[0]["fields"]
    return (fields.get("Video Topic") or "").strip(), fields.get("Storyboard Link") or ""


def fetch_storyboard_text(client: httpx.Client, storyboard_url: str) -> str:
    match = DOC_ID_RE.search(storyboard_url)
    if not match:
        raise RuntimeError(f"Invalid storyboard URL: {storyboard_url}")
    resp = client.get(
        f"https://docs.google.com/document/d/{match.group(1)}/export?format=txt",
        follow_redirects=True,
    )
    resp.raise_for_status()
    return resp.text


def split_speaker_segments(line: str) -> list[str]:
    matches = list(SPEAKER_RE.finditer(line))
    parts = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(line)
        segment = line[start:end]
        segment = segment.replace("|", " ")
        segment = normalize_storyboard_text(segment)
        if segment:
            parts.append(segment)
    return parts


def extract_primary_dialogue_block(doc_text: str) -> list[str]:
    lines = [line.rstrip() for line in doc_text.replace("\r", "").splitlines()]
    in_storyboard = False
    dialogue_lines: list[str] = []

    for raw in lines:
        line = raw.strip()
        if line == "STORYBOARD":
            if in_storyboard and dialogue_lines:
                break
            in_storyboard = True
            continue
        if not in_storyboard:
            continue
        if line.startswith("Key Takeaways:") or line.startswith("Pacing Guidance:") or line.startswith("Dialogue Notes:"):
            break
        if line.startswith("Topic:") and dialogue_lines:
            break
        if SPEAKER_RE.search(line):
            dialogue_lines.extend(split_speaker_segments(line))

    if not dialogue_lines:
        raise RuntimeError("Could not extract storyboard dialogue")
    return dialogue_lines


def parse_vtt(path: Path) -> list[Cue]:
    blocks = re.split(r"\n\s*\n", path.read_text(encoding="utf-8").strip())
    cues: list[Cue] = []
    for block in blocks:
        lines = [line.rstrip() for line in block.splitlines() if line.strip()]
        if not lines or lines[0] == "WEBVTT" or lines[0].startswith("Kind:") or lines[0].startswith("Language:"):
            continue
        time_index = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if time_index is None:
            continue
        start, end = [part.strip() for part in lines[time_index].split("-->")]
        text = "\n".join(lines[time_index + 1:]).strip()
        cues.append(Cue(start=start, end=end, text=text))
    return cues


def choose_breakpoint(text: str, start: int, target: int, remaining_chunks: int) -> int:
    remaining_text = text[start:].strip()
    if remaining_chunks <= 1:
        return len(text)

    min_remaining = (remaining_chunks - 1) * MIN_CHUNK_CHARS
    lower = max(start + MIN_CHUNK_CHARS, target - 20)
    upper = min(len(text) - min_remaining, target + 20)
    if upper <= lower:
        upper = min(len(text) - min_remaining, max(lower, target))

    candidates = []
    for i in range(lower, upper + 1):
        if i < len(text) and text[i].isspace():
            left = text[start:i].rstrip()
            if not left:
                continue
            score = abs(i - target)
            if re.search(r"[.!?]$", left):
                score -= 6
            elif re.search(r"[,;:]$", left):
                score -= 3
            candidates.append((score, i))

    if candidates:
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    fallback = min(len(text) - min_remaining, max(start + MIN_CHUNK_CHARS, target))
    while fallback < len(text) and not text[fallback].isspace():
        fallback += 1
    return min(fallback, len(text))


def distribute_text_across_cues(body_text: str, cues: list[Cue]) -> list[str]:
    body_text = normalize_storyboard_text(body_text)
    if not cues:
        return []
    total_duration = sum(max(cue.duration, 0.1) for cue in cues)
    total_chars = len(body_text)
    chunks: list[str] = []
    cursor = 0

    for idx, cue in enumerate(cues):
        remaining = len(cues) - idx
        if idx == len(cues) - 1:
            chunk = body_text[cursor:].strip()
            chunks.append(chunk)
            break

        share = max(cue.duration, 0.1) / total_duration
        target_chars = cursor + max(MIN_CHUNK_CHARS, int(round(total_chars * share)))
        breakpoint = choose_breakpoint(body_text, cursor, target_chars, remaining)
        chunk = body_text[cursor:breakpoint].strip()
        chunks.append(chunk)
        cursor = breakpoint

    return [format_chunk(chunk) for chunk in chunks]


def allocate_cue_counts(dialogue_lines: list[str], cue_count: int) -> list[int]:
    if cue_count <= 0:
        return []
    if cue_count <= len(dialogue_lines):
        counts = [0] * len(dialogue_lines)
        for i in range(cue_count):
            counts[i] = 1
        return counts

    lengths = [max(len(normalize_storyboard_text(line)), 1) for line in dialogue_lines]
    total = sum(lengths)
    counts = [1] * len(dialogue_lines)
    remaining = cue_count - len(dialogue_lines)

    while remaining > 0:
        best_idx = max(
            range(len(dialogue_lines)),
            key=lambda i: lengths[i] / counts[i],
        )
        counts[best_idx] += 1
        remaining -= 1

    return counts


def build_dialogue_cues(dialogue_lines: list[str], cues: list[Cue]) -> list[Cue]:
    counts = allocate_cue_counts(dialogue_lines, len(cues))
    rewritten: list[Cue] = []
    cue_index = 0

    for line, count in zip(dialogue_lines, counts):
        if count <= 0:
            continue
        cue_slice = cues[cue_index: cue_index + count]
        if not cue_slice:
            continue
        start_s = ts_to_seconds(cue_slice[0].start)
        end_s = ts_to_seconds(cue_slice[-1].end)
        total_duration = max(end_s - start_s, 0.3)
        chunks = split_subtitle_chunks(line)
        durations = allocate_chunk_durations(total_duration, chunks)
        cursor = start_s

        for idx, (chunk, duration) in enumerate(zip(chunks, durations)):
            chunk_end = end_s if idx == len(chunks) - 1 else min(end_s, cursor + duration)
            if chunk_end <= cursor:
                chunk_end = min(end_s, cursor + 0.3)
            rewritten.append(
                Cue(
                    start=seconds_to_ts(cursor),
                    end=seconds_to_ts(chunk_end),
                    text=chunk,
                )
            )
            cursor = chunk_end
        cue_index += count

    if cue_index < len(cues) and rewritten:
        rewritten[-1].end = cues[-1].end

    return rewritten


def format_chunk(chunk: str) -> str:
    chunk = normalize_storyboard_text(chunk)
    if len(chunk) <= MAX_LINE_CHARS:
        return chunk

    words = chunk.split()
    best = None
    for i in range(1, len(words)):
        left = " ".join(words[:i])
        right = " ".join(words[i:])
        longest = max(len(left), len(right))
        if longest > MAX_LINE_CHARS:
            continue
        score = abs(len(left) - len(right))
        if best is None or score < best[0]:
            best = (score, left, right)
    if best:
        return f"{best[1]}\n{best[2]}"
    return chunk


def split_subtitle_chunks(text: str) -> list[str]:
    text = normalize_storyboard_text(text)
    if not text:
        return []

    words = text.split()
    chunks: list[str] = []
    current: list[str] = []

    for word in words:
        candidate_words = current + [word]
        candidate_text = " ".join(candidate_words)

        if current and (len(candidate_text) > MAX_LINE_CHARS * 2 or len(candidate_words) > 16):
            chunks.append(" ".join(current))
            current = [word]
        else:
            current = candidate_words

        current_text = " ".join(current)
        if len(current_text) >= TARGET_CHUNK_CHARS and re.search(r"[.!?]$", current_text):
            chunks.append(current_text)
            current = []
        elif len(current_text) >= MAX_LINE_CHARS + 10 and re.search(r"[,;:]$", current_text):
            chunks.append(current_text)
            current = []

    if current:
        chunks.append(" ".join(current))

    chunks = [chunk for chunk in chunks if chunk.strip()]

    merged: list[str] = []
    for chunk in chunks:
        word_count = len(chunk.split())
        if merged and (word_count <= 2 or len(chunk) <= 12):
            merged[-1] = normalize_storyboard_text(f"{merged[-1]} {chunk}")
        else:
            merged.append(chunk)

    return [format_chunk(chunk) for chunk in merged if chunk.strip()]


def allocate_chunk_durations(total_duration: float, chunks: list[str]) -> list[float]:
    if not chunks:
        return []
    if len(chunks) == 1:
        return [total_duration]

    weights = [max(len(chunk.replace("\n", " ")), 1) for chunk in chunks]
    total_weight = sum(weights)
    floor = min(MIN_SUBTITLE_DURATION, total_duration / len(chunks))
    durations = [max(floor, total_duration * (weight / total_weight)) for weight in weights]
    allocated = sum(durations)

    if allocated > total_duration:
        scale = total_duration / allocated
        durations = [max(0.3, duration * scale) for duration in durations]
        allocated = sum(durations)

    if allocated < total_duration:
        durations[-1] += total_duration - allocated
    elif allocated > total_duration:
        durations[-1] = max(0.3, durations[-1] - (allocated - total_duration))

    return durations


def build_vtt(cues: list[Cue]) -> str:
    lines = ["WEBVTT", "Kind: captions", "Language: en", ""]
    for cue in cues:
        lines.append(f"{cue.start} --> {cue.end}")
        lines.append(cue.text)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def find_media_path(vtt_path: Path) -> Path | None:
    for suffix in [".mp4", ".mov", ".m4v", ".webm", ".mkv"]:
        candidate = vtt_path.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return None


def transcribe_media_tail(media_path: Path, prompt: str | None = None) -> tuple[float, float, str] | None:
    if not os.environ.get("OPENAI_API_KEY"):
        return None

    duration = run_ffprobe_duration(media_path)
    start_offset = max(duration - TAIL_TRANSCRIBE_SECONDS, 0.0)
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-sseof",
                f"-{TAIL_TRANSCRIBE_SECONDS}",
                "-i",
                str(media_path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                str(tmp_path),
                "-y",
            ],
            check=True,
            capture_output=True,
        )
        client = OpenAI()
        with open(tmp_path, "rb") as f:
            kwargs = {
                "model": "whisper-1",
                "file": f,
                "response_format": "verbose_json",
            }
            if prompt:
                kwargs["prompt"] = prompt[:900]
            result = client.audio.transcriptions.create(**kwargs)
        text = normalize_storyboard_text(getattr(result, "text", "") or "")
        return (duration, start_offset, text) if text else None
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass


def run_ffprobe_duration(media_path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(media_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def extract_novel_suffix(existing_text: str, tail_text: str) -> str:
    existing_words = normalize_storyboard_text(existing_text).split()
    tail_words = normalize_storyboard_text(tail_text).split()
    max_overlap = 0
    limit = min(len(existing_words), len(tail_words))

    for overlap in range(1, limit + 1):
        if existing_words[-overlap:] == tail_words[:overlap]:
            max_overlap = overlap

    novel_words = tail_words[max_overlap:]
    return " ".join(novel_words).strip()


def clean_tail_suffix(text: str) -> str:
    text = normalize_storyboard_text(text)
    if not text:
        return text

    sentences = re.split(r"(?<=[.!?])\s+", text)
    cleaned: list[str] = []
    seen_counts: dict[str, int] = {}

    for sentence in sentences:
        sentence = normalize_storyboard_text(sentence)
        if not sentence:
            continue
        key = sentence.lower()
        seen_counts[key] = seen_counts.get(key, 0) + 1
        if seen_counts[key] > 1:
            continue
        cleaned.append(sentence)

    text = " ".join(cleaned)
    text = re.sub(r"\b(.{1,30}?)\b(?:\s+\1\b){2,}", r"\1", text, flags=re.IGNORECASE)
    return normalize_storyboard_text(text)


def append_tail_transcript(cues: list[Cue], media_path: Path) -> list[Cue]:
    tail = transcribe_media_tail(media_path, prompt=" ".join(cue.text.replace("\n", " ") for cue in cues[-4:]))
    if not tail:
        return cues

    video_end_s, tail_start_s, tail_text = tail
    last_end_s = ts_to_seconds(cues[-1].end)
    if video_end_s - last_end_s < 2.0:
        return cues

    existing_tail_text = " ".join(cue.text.replace("\n", " ") for cue in cues if ts_to_seconds(cue.start) >= tail_start_s)
    novel_suffix = clean_tail_suffix(extract_novel_suffix(existing_tail_text, tail_text))
    if len(novel_suffix.split()) < 4:
        return cues

    tail_chunks = split_subtitle_chunks(novel_suffix)
    total_duration = max(video_end_s - last_end_s, 0.3)
    durations = allocate_chunk_durations(total_duration, tail_chunks)
    start_cursor = last_end_s
    appended: list[Cue] = []
    for idx, (chunk, duration) in enumerate(zip(tail_chunks, durations)):
        end_time = video_end_s if idx == len(tail_chunks) - 1 else start_cursor + duration
        appended.append(
            Cue(
                start=seconds_to_ts(start_cursor),
                end=seconds_to_ts(end_time),
                text=chunk,
            )
        )
        start_cursor = end_time

    return cues + appended


def finalize_file(client: httpx.Client, path: Path) -> tuple[str, int, int]:
    video_id = extract_video_id(path.name)
    if video_id is None:
        raise RuntimeError(f"No video id in filename: {path.name}")
    if video_id in SKIP_VIDEO_IDS:
        return ("skipped", 0, 0)

    topic, storyboard_url = fetch_storyboard_link(client, video_id)
    storyboard_text = fetch_storyboard_text(client, storyboard_url)
    dialogue_lines = extract_primary_dialogue_block(storyboard_text)

    cues = parse_vtt(path)
    if not cues:
        raise RuntimeError(f"No cues found in {path.name}")

    preserved = []
    rewrite_from = 0
    if "riptoes" in cues[0].text.lower():
        preserved.append(cues[0])
        rewrite_from = 1

    rewritten_cues = preserved + build_dialogue_cues(dialogue_lines, cues[rewrite_from:])
    media_path = find_media_path(path)
    if media_path is not None:
        rewritten_cues = append_tail_transcript(rewritten_cues, media_path)

    path.write_text(build_vtt(rewritten_cues), encoding="utf-8")
    return (topic or path.stem, len(cues), len(dialogue_lines))


def main() -> None:
    if not os.environ.get("AIRTABLE_TOKEN"):
        raise SystemExit("AIRTABLE_TOKEN is required")

    files = sorted(VTT_DIR.glob("*.vtt"))
    with httpx.Client(timeout=40) as client:
        for path in files:
            video_id = extract_video_id(path.name)
            if video_id in SKIP_VIDEO_IDS:
                print(f"SKIP {path.name}")
                continue
            topic, cue_count, dialogue_count = finalize_file(client, path)
            print(f"OK {path.name} | cues={cue_count} | dialogue_lines={dialogue_count} | topic={topic}")


if __name__ == "__main__":
    main()
