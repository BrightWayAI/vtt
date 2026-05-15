import os
import re
from pathlib import Path

import httpx

from finalize_vtts_from_storyboards import (
    VTT_DIR,
    extract_video_id,
    fetch_storyboard_link,
    fetch_storyboard_text,
    extract_primary_dialogue_block,
    normalize_storyboard_text,
    split_subtitle_chunks,
    allocate_chunk_durations,
    seconds_to_ts,
    ts_to_seconds,
)


GAP_THRESHOLD = 10.0


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


def norm(text: str) -> str:
    text = normalize_storyboard_text(text).lower()
    text = re.sub(r"[^a-z0-9\s']+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def find_gap(cues: list[tuple[str, str, str]]) -> tuple[int, float, float] | None:
    prev_end = None
    for idx, (start, end, text) in enumerate(cues):
        start_s = ts_to_seconds(start)
        if prev_end is not None and start_s - prev_end > GAP_THRESHOLD:
            return idx, prev_end, start_s
        prev_end = ts_to_seconds(end)
    return None


def infer_used_line_count(before_text: str, dialogue_lines: list[str]) -> int:
    before = norm(before_text)
    used = 0
    for line in dialogue_lines:
        if norm(line) and norm(line) in before:
            used += 1
        else:
            break
    return used


def synthesize_gap_cues(lines: list[str], start_s: float, end_s: float) -> list[tuple[str, str, str]]:
    if not lines or end_s <= start_s:
        return []
    chunks: list[str] = []
    for line in lines:
        chunks.extend(split_subtitle_chunks(line))
    durations = allocate_chunk_durations(end_s - start_s, chunks)
    cues = []
    cursor = start_s
    for i, (chunk, duration) in enumerate(zip(chunks, durations)):
        cue_end = end_s if i == len(chunks) - 1 else min(end_s, cursor + duration)
        cues.append((seconds_to_ts(cursor), seconds_to_ts(cue_end), chunk))
        cursor = cue_end
    return cues


def main() -> None:
    if not os.environ.get("AIRTABLE_TOKEN"):
        raise SystemExit("AIRTABLE_TOKEN is required")

    with httpx.Client(timeout=40) as client:
        for path in sorted(VTT_DIR.glob("*.vtt")):
            video_id = extract_video_id(path.name)
            if video_id == 30:
                continue
            cues = parse_vtt(path)
            gap = find_gap(cues)
            if not gap:
                continue
            idx, gap_start, gap_end = gap
            before_text = " ".join(text.replace("\n", " ") for _, end, text in cues if ts_to_seconds(end) <= gap_start)
            _, storyboard_url = fetch_storyboard_link(client, video_id)
            dialogue_lines = extract_primary_dialogue_block(fetch_storyboard_text(client, storyboard_url))
            used = infer_used_line_count(before_text, dialogue_lines)
            gap_lines = dialogue_lines[used:]
            if not gap_lines:
                continue
            inserted = synthesize_gap_cues(gap_lines, gap_start, gap_end)
            new_cues = cues[:idx] + inserted + cues[idx:]
            path.write_text(build_vtt(new_cues), encoding="utf-8")
            print(f"REPAIRED {path.name} | inserted={len(inserted)}")


if __name__ == "__main__":
    main()
