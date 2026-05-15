import re
from dataclasses import dataclass
from pathlib import Path

from finalize_vtts_from_storyboards import (
    VTT_DIR,
    allocate_chunk_durations,
    format_chunk,
    seconds_to_ts,
    split_subtitle_chunks,
    ts_to_seconds,
)


@dataclass
class Cue:
    start: str
    end: str
    text: str

    @property
    def start_s(self) -> float:
        return ts_to_seconds(self.start)

    @property
    def end_s(self) -> float:
        return ts_to_seconds(self.end)


TAIL_REPLACEMENTS: dict[str, dict[str, object]] = {
    "25_John Muir and the Conservation Movement_Final.vtt": {
        "start": 84.586,
        "lines": [
            "He asked the government to make it a national park.",
            "That's right. Today, families still visit Yosemite.",
            "They see the same tall trees and waterfalls Muir wrote about long ago.",
            "Because it's a national park, those trees are still protected and can't be cut down.",
            "It's really cool that one person helped protect a whole forest.",
            "And now, since it's a national park, it's still there for everyone to visit.",
            "Is there a place you'd want to protect?",
        ],
    },
    "53_Franklin D. Roosevelt and the New Deal_Final.vtt": {
        "start": 96.792,
        "lines": [
            "During the Great Depression, President Roosevelt created the New Deal to put people to work.",
            "They built roads, bridges, parks, and schools, things the whole country needed.",
            "And we still use what they built today, almost a hundred years later.",
            "If you could build something to help your community, what would you make?",
        ],
    },
    "98_Exploring America’s Islands and Territories_Final.vtt": {
        "start": 45.227,
        "lines": [
            "They also send representatives to Congress in Washington, D.C., to share their voices.",
            "What do you think might feel different about living in a U.S. territory instead of a state?",
            "Let's visit some U.S. territories!",
            "Puerto Rico and the U.S. Virgin Islands are in the warm Caribbean Sea.",
            "Families in Puerto Rico speak Spanish or English, keeping their island's long history alive.",
            "Other territories are far across the Pacific Ocean.",
            "Guam and American Samoa have ancient histories that existed thousands of years before joining America.",
            "Each territory has its own history and leaders, and each one is still an important part of the United States.",
            "I didn't know there were parts of the United States that are so far away.",
            "Or that they could be so different.",
            "I want to learn more about their history.",
            "What piece of history or tradition makes your community special to you?",
        ],
    },
    "107_The Dust Bowl_Final.vtt": {
        "start": 59.740,
        "lines": [
            "Grass and plant roots hold soil in place, like fingers gripping the ground.",
            "Now farmers plant cover crops, extra plants that cover the ground and protect it.",
            "Their roots hold the soil down tight so wind can't blow it away.",
            "And scientists watch the weather and warn farmers when a drought is coming.",
            "That way, everyone can get ready.",
            "The Dust Bowl was really hard for those families.",
            "But people learned from it. Now they know how to take care of their land.",
            "Planting grass and cover crops keeps the soil safe.",
            "What would you like to grow in your own garden?",
        ],
    },
    "116_The Story of Route 66_Final.vtt": {
        "start": 65.300,
        "lines": [
            "If I were on Route 66, I'd want to stop at every single town and get a milkshake at every diner.",
            "Maybe we can take a road trip this summer.",
            "If you could take a road trip, where would you stop first?",
        ],
    },
    "144_Marian Anderson – Opera Singer_Final.vtt": {
        "start": 91.096,
        "lines": [
            "She worked on her voice every day and learned to sing in different languages.",
            "And people believed in her and helped her get lessons.",
            "In 1955, Marian sang at the Metropolitan Opera in New York City, becoming the first African American to perform on that stage.",
            "I like that her whole church helped her keep singing.",
            "That's a pretty big jump from a little church to the whole world.",
            "All that practice paid off.",
            "What is a talent you have that you would enjoy sharing with the world?",
        ],
    },
    "164_The Founding of the Girl Scouts_Final.vtt": {
        "start": 83.959,
        "lines": [
            "Today, Girl Scouts across the country still earn badges and complete service projects.",
            "There are over 2 million Girl Scout members, including adult members, across the United States.",
            "They are part of a global organization in more than 100 countries.",
            "They were learning Morse code and first aid? That's so cool!",
            "I'm glad they didn't just stay inside.",
            "Yeah, they actually got to try new things!",
            "What's something you've always wanted to try?",
        ],
    },
    "198_The Grand Canyon_Final.vtt": {
        "start": 139.521,
        "lines": [
            "If you dug a deep hole in your neighborhood, what do you think you'd find?",
        ],
    },
    "201_The Panama Canal_Final.vtt": {
        "start": 103.353,
        "lines": [
            "The Panama Canal opened in 1914.",
            "Today, thousands of ships use it every year to share food and goods around the world.",
            "What an amazing invention! The water does all the lifting!",
            "And it almost didn't happen because of a bunch of mosquitoes.",
            "I'm glad they figured it out.",
            "Now it only takes a day to get across instead of months.",
            "If you could build a shortcut to anywhere, where would you want to go?",
        ],
    },
}


def parse_vtt(path: Path) -> list[Cue]:
    blocks = re.split(r"\n\s*\n", path.read_text(encoding="utf-8").strip())
    cues: list[Cue] = []
    for block in blocks:
        lines = [line.rstrip() for line in block.splitlines() if line.strip()]
        if not lines or lines[0] == "WEBVTT" or lines[0].startswith("Kind:") or lines[0].startswith("Language:"):
            continue
        idx = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if idx is None:
            continue
        start, end = [part.strip() for part in lines[idx].split("-->")]
        text = "\n".join(lines[idx + 1:]).strip()
        cues.append(Cue(start=start, end=end, text=text))
    return cues


def build_vtt(cues: list[Cue]) -> str:
    lines = ["WEBVTT", "Kind: captions", "Language: en", ""]
    for cue in cues:
        lines.append(f"{cue.start} --> {cue.end}")
        lines.append(cue.text)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def flatten_chunks(lines: list[str]) -> list[str]:
    chunks: list[str] = []
    for line in lines:
        split = split_subtitle_chunks(line)
        if split:
            chunks.extend(split)
    return [format_chunk(chunk) for chunk in chunks if chunk.strip()]


def find_replace_index(cues: list[Cue], start_s: float) -> int:
    for idx, cue in enumerate(cues):
        if cue.start_s >= start_s - 0.01 or cue.end_s > start_s:
            return idx
    return len(cues)


def rebuild_range(start_s: float, end_s: float, lines: list[str]) -> list[Cue]:
    chunks = flatten_chunks(lines)
    if not chunks:
        return []
    durations = allocate_chunk_durations(max(end_s - start_s, 0.3), chunks)
    rebuilt: list[Cue] = []
    cursor = start_s
    for idx, (chunk, duration) in enumerate(zip(chunks, durations)):
        cue_end = end_s if idx == len(chunks) - 1 else min(end_s, cursor + duration)
        if cue_end <= cursor:
            cue_end = min(end_s, cursor + 0.3)
        rebuilt.append(Cue(start=seconds_to_ts(cursor), end=seconds_to_ts(cue_end), text=chunk))
        cursor = cue_end
    return rebuilt


def replace_tail(cues: list[Cue], start_s: float, lines: list[str]) -> list[Cue]:
    idx = find_replace_index(cues, start_s)
    if idx >= len(cues):
        return cues
    end_s = cues[-1].end_s
    preserved = cues[:idx]
    rebuilt = rebuild_range(start_s, end_s, lines)
    return preserved + rebuilt


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", text))


def merge_adjacent_text(left: str, right: str) -> str:
    left = left.replace("\n", " ").strip()
    right = right.replace("\n", " ").strip()
    if not left:
        return format_chunk(right)
    if not right:
        return format_chunk(left)
    return format_chunk(f"{left} {right}")


def cleanup_orphan_cues(cues: list[Cue]) -> list[Cue]:
    cleaned: list[Cue] = []
    for cue in cues:
        text = cue.text.replace("\n", " ").strip()
        if (
            cleaned
            and word_count(text) <= 2
            and len(text) <= 16
            and not cleaned[-1].text.replace("\n", " ").strip().endswith((".", "!", "?"))
        ):
            cleaned[-1].text = merge_adjacent_text(cleaned[-1].text, cue.text)
            cleaned[-1].end = cue.end
            continue
        cleaned.append(cue)
    return cleaned


def find_orphans(cues: list[Cue]) -> list[Cue]:
    return [
        cue
        for cue in cues
        if word_count(cue.text.replace("\n", " ").strip()) <= 2 and len(cue.text.replace("\n", " ").strip()) <= 16
    ]


def find_large_gaps(cues: list[Cue], threshold: float = 6.0) -> list[tuple[float, float]]:
    gaps: list[tuple[float, float]] = []
    prev_end = None
    for cue in cues:
        if prev_end is not None and cue.start_s - prev_end > threshold:
            gaps.append((prev_end, cue.start_s))
        prev_end = cue.end_s
    return gaps


def main() -> None:
    for path in sorted(VTT_DIR.glob("*.vtt")):
        if path.name == "30_Thomas Edison and the Light Bulb_Final.vtt":
            print(f"SKIP {path.name}")
            continue

        cues = parse_vtt(path)
        replacement = TAIL_REPLACEMENTS.get(path.name)
        if replacement:
            cues = replace_tail(cues, float(replacement["start"]), list(replacement["lines"]))

        cues = cleanup_orphan_cues(cues)
        path.write_text(build_vtt(cues), encoding="utf-8")

        orphan_count = len(find_orphans(cues))
        gaps = find_large_gaps(cues)
        print(f"OK {path.name} | orphans={orphan_count} | gaps={gaps}")


if __name__ == "__main__":
    main()
