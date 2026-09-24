"""Stable phrase captions built from acoustic word boundaries, never karaoke cues."""
import html
import math
import re

MAX_LINE = 42
MAX_CHARS = 84
MAX_WORDS = 16
MAX_DURATION = 6.0
MIN_DURATION = 1.2
TARGET_CPS = 17.0
PAUSE = 0.6


def tokens(text):
    return re.findall(r"\w+(?:['’]\w+)*", text.casefold().replace('’', "'"))


def wrap(text):
    words = text.split()
    if len(text) <= MAX_LINE:
        return text
    candidates = [(" ".join(words[:i]), " ".join(words[i:])) for i in range(1, len(words))]
    if not candidates:
        return text
    left, right = min(candidates, key=lambda p: (max(0, max(map(len, p)) - MAX_LINE), abs(len(p[0]) - len(p[1]))))
    return left + '\n' + right


def fits(text):
    return len(text) <= MAX_CHARS and len(text.split()) <= MAX_WORDS and all(len(line) <= MAX_LINE for line in wrap(text).splitlines())


def phrase_cues(words, segments, duration):
    """Use words only when they reproduce the segment; otherwise preserve its text.

    Assign each acoustic word to a single segment. Segment-only fallback is explicitly
    approximate; it must never silently drop text because word counts differ.
    """
    groups = [[] for _ in segments]
    for word in words:
        start, end = float(word['start']), float(word['end'])
        if not math.isfinite(start) or not math.isfinite(end) or end < start:
            raise ValueError('Invalid word timestamps')
        midpoint = (start + end) / 2
        candidates = [i for i, seg in enumerate(segments) if seg['start'] - .05 <= midpoint <= seg['end'] + .05]
        if candidates:
            i = max(candidates, key=lambda i: min(end, segments[i]['end']) - max(start, segments[i]['start']))
            groups[i].append(word)

    units = []
    for seg, group in zip(segments, groups):
        text = ' '.join(seg.get('text', '').split())
        if not text:
            continue
        if group and tokens(' '.join(w['word'] for w in group)) == tokens(text) and len(text.split()) == len(group):
            units.extend({'text': token, 'start': w['start'], 'end': w['end']}
                         for token, w in zip(text.split(), group))
        else:
            # Missing or incompatible word metadata: keep every recognized token.
            parts = text.split()
            weights = [len(p) for p in parts]
            total = sum(weights)
            cursor = seg['start']
            for token, weight in zip(parts, weights):
                end = cursor + (seg['end'] - seg['start']) * weight / total
                units.append({'text': token, 'start': cursor, 'end': end})
                cursor = end

    cues = []
    current = None
    for unit in units:
        start, end = max(0., unit['start']), min(duration, unit['end'])
        if start >= duration or end < start:
            raise ValueError('Speech timestamps outside media duration')
        candidate = (current['text'] + ' ' if current else '') + unit['text']
        split = current and (
            not fits(candidate) or end - current['start'] > MAX_DURATION
            or start - current['end'] > PAUSE
            or (re.search(r'[.!?][\"\']?$', current['text'])
                and current['end'] - current['start'] >= MIN_DURATION)
        )
        if split:
            cues.append(current)
            current = None
        if current is None:
            current = {'start': start, 'end': end, 'text': unit['text'], '_units': [unit]}
        else:
            current['_units'].append(unit)
            current['text'] = candidate
            current['end'] = max(current['end'], end)
    if current:
        cues.append(current)

    # Rebalance a length split so a sentence does not leave a one-word flash.
    for previous, cue in zip(cues, cues[1:]):
        while (len(cue['_units']) < 3 or cue['end'] - cue['start'] < MIN_DURATION):
            if len(previous['_units']) <= 3 or cue['start'] - previous['end'] > PAUSE:
                break
            moved = previous['_units'][-1]
            text = moved['text'] + ' ' + cue['text']
            if (not fits(text) or cue['end'] - moved['start'] > MAX_DURATION
                or previous['_units'][-2]['end'] - previous['start'] < MIN_DURATION
                or re.search(r'[.!?][\"\']?$', previous['text'])):
                break
            previous['_units'].pop()
            previous['text'] = ' '.join(u['text'] for u in previous['_units'])
            previous['end'] = previous['_units'][-1]['end']
            cue['_units'].insert(0, moved)
            cue.update(start=moved['start'], text=text)
    for cue in cues:
        del cue['_units']

    # Merge a short final phrase where it fits, without bridging a real pause.
    merged = []
    for cue in cues:
        if merged:
            prev = merged[-1]
            text = prev['text'] + ' ' + cue['text']
            if (cue['end'] - cue['start'] < MIN_DURATION
                and cue['start'] - prev['end'] <= PAUSE
                and cue['end'] - prev['start'] <= MAX_DURATION and fits(text)):
                prev.update(text=text, end=cue['end'])
                continue
        merged.append(dict(cue))
    for i, cue in enumerate(merged):
        next_start = merged[i + 1]['start'] if i + 1 < len(merged) else duration
        # Only borrow a little silence. Never delay following speech to fit a cue.
        desired = cue['start'] + max(MIN_DURATION, len(cue['text']) / TARGET_CPS)
        cue['end'] = min(next_start, duration, cue['start'] + MAX_DURATION,
                         max(cue['end'], min(desired, cue['end'] + .75)))
        if cue['end'] <= cue['start']:
            raise ValueError('Cannot render speech with non-increasing timestamps')
    return merged


def quality_report(cues):
    """Readability warnings, not a claim that the transcript matches the audio."""
    return {
        'cue_count': len(cues),
        'short_cues': sum(c['end'] - c['start'] < MIN_DURATION - .001 for c in cues),
        'fast_cues': sum(len(c['text']) / max(.001, c['end'] - c['start']) > 22 for c in cues),
        'long_lines': sum(any(len(line) > MAX_LINE for line in wrap(c['text']).splitlines()) for c in cues),
    }


def render(words, segments, duration, timestamp):
    cues = phrase_cues(words, segments, duration)
    lines = ['WEBVTT', 'Kind: captions', 'Language: en', '']
    for cue in cues:
        lines += [f"{timestamp(cue['start'])} --> {timestamp(cue['end'])}",
                  html.escape(wrap(cue['text']), quote=False), '']
    return '\n'.join(lines), quality_report(cues)
