import os
import re
import subprocess
import tempfile
from pathlib import Path

from openai import OpenAI


VTT_DIR = Path.home() / "Desktop" / "Videos for Upload"
SKIP_VIDEO_IDS = {30}
TAIL_SECONDS = 35
MEDIA_SUFFIXES = [".mp4", ".mov", ".m4v", ".webm", ".mkv"]


def extract_video_id(filename: str) -> int | None:
    match = re.match(r"^(\d+)[_\s]", Path(filename).stem)
    return int(match.group(1)) if match else None


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


def ts_to_seconds(ts: str) -> float:
    h, m, rest = ts.split(":")
    s, ms = rest.split(".")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def normalize(text: str) -> str:
    text = text.replace("\ufeff", "")
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2014", "-").replace("\u2013", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def format_chunk(text: str) -> str:
    text = normalize(text)
    if len(text) <= 45:
        return text
    words = text.split()
    best = None
    for i in range(1, len(words)):
        left = " ".join(words[:i])
        right = " ".join(words[i:])
        longest = max(len(left), len(right))
        if longest > 45:
            continue
        score = abs(len(left) - len(right))
        if best is None or score < best[0]:
            best = (score, left, right)
    if best:
        return f"{best[1]}\n{best[2]}"
    return text


def find_media_path(vtt_path: Path) -> Path | None:
    for suffix in MEDIA_SUFFIXES:
        candidate = vtt_path.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return None


def ffprobe_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def parse_vtt(path: Path) -> list[tuple[str, str, str]]:
    blocks = re.split(r"\n\s*\n", path.read_text(encoding="utf-8").strip())
    cues = []
    for block in blocks:
        lines = [line.rstrip() for line in block.splitlines() if line.strip()]
        if not lines or lines[0] == "WEBVTT" or lines[0].startswith("Kind:") or lines[0].startswith("Language:"):
            continue
        idx = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if idx is None:
            continue
        start, end = [part.strip() for part in lines[idx].split("-->")]
        text = "\n".join(lines[idx + 1:]).strip()
        cues.append((start, end, text))
    return cues


def build_vtt(cues: list[tuple[str, str, str]]) -> str:
    lines = ["WEBVTT", "Kind: captions", "Language: en", ""]
    for start, end, text in cues:
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def transcribe_tail(media_path: Path, prompt: str | None = None) -> tuple[float, list[tuple[str, str, str]]]:
    duration = ffprobe_duration(media_path)
    tail_start = max(duration - TAIL_SECONDS, 0.0)
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-sseof",
                f"-{TAIL_SECONDS}",
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
                "timestamp_granularities": ["segment"],
            }
            if prompt:
                kwargs["prompt"] = prompt[:900]
            result = client.audio.transcriptions.create(**kwargs)
        cues = []
        for seg in getattr(result, "segments", None) or []:
            start = tail_start + float(seg.start)
            end = tail_start + float(seg.end)
            text = format_chunk(getattr(seg, "text", ""))
            if text:
                cues.append((seconds_to_ts(start), seconds_to_ts(end), text))
        return tail_start, cues
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required")

    for vtt_path in sorted(VTT_DIR.glob("*.vtt")):
        video_id = extract_video_id(vtt_path.name)
        if video_id in SKIP_VIDEO_IDS:
            print(f"SKIP {vtt_path.name}")
            continue
        media_path = find_media_path(vtt_path)
        if media_path is None:
            print(f"NO_MEDIA {vtt_path.name}")
            continue

        cues = parse_vtt(vtt_path)
        prompt = " ".join(text.replace("\n", " ") for _, _, text in cues[-4:])
        tail_start, tail_cues = transcribe_tail(media_path, prompt=prompt)
        kept = [(s, e, t) for s, e, t in cues if ts_to_seconds(e) <= tail_start]
        final_cues = kept + tail_cues
        vtt_path.write_text(build_vtt(final_cues), encoding="utf-8")
        print(f"OK {vtt_path.name} | kept={len(kept)} | tail={len(tail_cues)}")


if __name__ == "__main__":
    main()
