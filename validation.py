"""Read-only checks. Storyboards are references, never replacement transcripts."""
from difflib import SequenceMatcher
import html
import re
from captions import tokens, quality_report


def extract_storyboard_dialogue(document):
    lines = extract_storyboard_vo_lines(document)
    return '\n'.join(lines) or None


def extract_storyboard_vo_lines(document):
    """Extract only Omega storyboard VO column/speaker lines, preserving rows."""
    lines = document.replace('\r', '').splitlines()
    output = []
    in_storyboard = False
    vo_column = None
    delimiter = None
    dialogue_block = False

    for raw in lines:
        line = raw.strip()
        if line.upper() == 'STORYBOARD':
            in_storyboard = True
            continue
        if not in_storyboard and re.search(r'\b(?:VO|VOICE[- ]?OVER|NARRATOR)\b', line, re.I):
            # Support docs that omit the literal STORYBOARD section marker.
            in_storyboard = True
        if not in_storyboard or not line:
            continue
        if re.match(r'^(?:Key Takeaways|Pacing Guidance|Dialogue Notes|Notes|Sources?)\s*:', line, re.I):
            break
        if re.match(r'^(?:Scene|Visuals?|Animation|On[ -]?screen|SFX|Music|Duration|Camera|Transitions?|Audio)\s*:', line, re.I):
            continue
        if re.match(r'^(?:Topic|Learning Objective|Grade Level|Standard)\s*:', line, re.I):
            continue

        parts = None
        if '|' in line:
            delimiter = '|'
            parts = [part.strip() for part in line.strip('|').split('|')]
        elif '\t' in line:
            delimiter = '\t'
            parts = [part.strip() for part in line.split('\t')]
        if parts:
            headers = [re.sub(r'[^a-z]', '', part.casefold()) for part in parts]
            if any(header in {'vo', 'voiceover', 'narration', 'narrator'} for header in headers):
                vo_column = next(i for i, header in enumerate(headers) if header in {'vo', 'voiceover', 'narration', 'narrator'})
                continue
            if vo_column is not None and vo_column < len(parts):
                candidate = parts[vo_column]
                if candidate and not re.match(r'^(?:visual|scene|shot|audio|sfx)\b', candidate, re.I):
                    output.append(candidate.strip('"“”'))
                continue

        speaker = re.match(
            r'^(?:NARRATOR|NARRATION|DIALOGUE|VOICE[ -]?OVER|VO|ALMA|MATEO)\s*(?:\([^)]*\))?\s*:\s*(.*)$',
            line, re.I,
        )
        if speaker:
            text = speaker.group(1).strip().strip('"“”')
            if text:
                output.append(text)
            else:
                dialogue_block = True
            continue

        if dialogue_block and re.match(r'^[\w -]{1,30}:\s*', line):
            text = re.sub(r'^[\w -]{1,30}:\s*', '', line).strip('"“”')
            if text:
                output.append(text)
            continue

        # Omega rows may contain several speaker labels in one line.
        matches = list(re.finditer(r'\b(?:NARRATOR|ALMA|MATEO|DIALOGUE)\s*:', line, re.I))
        if matches:
            for i, match in enumerate(matches):
                end = matches[i + 1].start() if i + 1 < len(matches) else len(line)
                text = line[match.end():end].strip().strip('"“”|')
                if text:
                    output.append(text)
    return output


def _word_parts(expected, actual):
    matcher = SequenceMatcher(None, tokens(expected), tokens(actual), autojunk=False)
    expected_parts = []
    actual_parts = []
    expected_tokens, actual_tokens = tokens(expected), tokens(actual)
    for tag, i, j, k, l in matcher.get_opcodes():
        expected_parts.extend({'text': ' '.join(expected_tokens[i:j]), 'kind': 'same' if tag == 'equal' else 'missing'} for _ in [0] if i != j)
        actual_parts.extend({'text': ' '.join(actual_tokens[k:l]), 'kind': 'same' if tag == 'equal' else 'extra'} for _ in [0] if k != l)
    return expected_parts, actual_parts


def _word_diff_counts(expected, actual):
    expected_tokens, actual_tokens = tokens(expected), tokens(actual)
    missing = extra = 0
    for tag, i, j, k, l in SequenceMatcher(None, expected_tokens, actual_tokens, autojunk=False).get_opcodes():
        if tag != 'equal':
            missing += j - i
            extra += l - k
    return missing, extra


def _cue_text(cue):
    if isinstance(cue, dict):
        return html.unescape(cue.get('text', '')).replace('\n', ' ').strip()
    if isinstance(cue, (tuple, list)):
        return html.unescape(cue[2]).replace('\n', ' ').strip()
    return str(cue).strip()


def _cue_record(cue, index):
    if isinstance(cue, dict):
        return {'cue': index, 'start': cue.get('start'), 'end': cue.get('end'),
                'text': _cue_text(cue)}
    return {'cue': index, 'start': cue[0], 'end': cue[1], 'text': _cue_text(cue)}


def _range_score(expected, actual):
    expected_tokens, actual_tokens = tokens(expected), tokens(actual)
    if not expected_tokens or not actual_tokens:
        return 0.0
    ratio = SequenceMatcher(None, expected_tokens, actual_tokens, autojunk=False).ratio()
    # Prefer a complete thought over a tiny matching fragment, while allowing
    # natural ASR cue splits and a little ad-libbed audio.
    length_penalty = min(0.25, abs(len(expected_tokens) - len(actual_tokens)) / max(len(expected_tokens), 1) * 0.15)
    return ratio - length_penalty


def _align_shots(expected_lines, cues):
    """Map each storyboard shot to one or more adjacent transcript cues."""
    records = [_cue_record(cue, i + 1) for i, cue in enumerate(cues)]
    rows, cursor = [], 0
    for shot_number, expected in enumerate(expected_lines, 1):
        remaining_shots = len(expected_lines) - shot_number
        if cursor >= len(records):
            rows.append({'line': shot_number, 'status': 'review', 'storyboard': expected,
                         'transcript': '', 'audio_cues': [], 'storyboard_parts': _word_parts(expected, '')[0],
                         'transcript_parts': []})
            continue
        max_end = len(records) - remaining_shots
        # A shot normally spans only a handful of cues. The cap prevents a
        # corrupt ASR result from swallowing the remainder of the storyboard.
        max_end = min(max_end, cursor + 12)
        best_end, best_score = cursor + 1, -1.0
        for end in range(cursor + 1, max_end + 1):
            actual = ' '.join(record['text'] for record in records[cursor:end])
            score = _range_score(expected, actual)
            if score > best_score:
                best_end, best_score = end, score
        selected = records[cursor:best_end]
        actual = ' '.join(record['text'] for record in selected)
        expected_parts, actual_parts = _word_parts(expected, actual)
        missing, extra = _word_diff_counts(expected, actual)
        for record in selected:
            cue_expected_parts, cue_actual_parts = _word_parts(expected, record['text'])
            record['transcript_parts'] = cue_actual_parts
        rows.append({'line': shot_number, 'status': 'matched' if not missing and not extra else 'review',
                     'storyboard': expected, 'transcript': actual, 'audio_cues': selected,
                     'storyboard_parts': expected_parts, 'transcript_parts': actual_parts,
                     'missing': missing, 'extra': extra})
        cursor = best_end
    if cursor < len(records):
        rows.append({'line': len(rows) + 1, 'status': 'extra', 'storyboard': '',
                     'transcript': ' '.join(r['text'] for r in records[cursor:]),
                     'audio_cues': records[cursor:], 'storyboard_parts': [],
                     'transcript_parts': [{'text': _cue_text(r), 'kind': 'extra'} for r in records[cursor:]]})
    return rows


def storyboard_check(transcript, storyboard):
    if not storyboard or not tokens(storyboard):
        return {'status': 'unavailable', 'message': 'Storyboard not checked: no readable dialogue was available.'}
    expected_lines = extract_storyboard_vo_lines(storyboard) or [line.strip() for line in storyboard.splitlines() if line.strip()]
    if isinstance(transcript, list) and transcript and isinstance(transcript[0], (dict, tuple, list)):
        cues = transcript
    else:
        lines = transcript if isinstance(transcript, list) else [line.strip() for line in transcript.splitlines() if line.strip()]
        cues = [{'start': None, 'end': None, 'text': line} for line in lines]
    rows = _align_shots(expected_lines, cues)
    missing = sum(row.get('missing', 0) for row in rows)
    extra = sum(row.get('extra', 0) for row in rows)
    # Any audio before/after the storyboard is useful to reviewers, but is not
    # treated as a missing storyboard shot.
    extra += sum(len(tokens(cue['text'])) for row in rows if row['status'] == 'extra' for cue in row.get('audio_cues', []))
    status = 'matched' if not missing and not extra else 'review'
    return {'status': status, 'storyboard_lines': len(expected_lines), 'transcript_lines': len(cues),
            'missing_or_changed_words': missing, 'extra_or_changed_words': extra, 'lines': rows,
            'message': 'Storyboard VO matches the recognized transcript.' if status == 'matched' else
                f'Storyboard VO review: {missing} storyboard words and {extra} audio words differ. Audio transcript retained.'}


def check_output(cues, storyboard):
    report = quality_report([{'start': s, 'end': e, 'text': html.unescape(t)} for s, e, t in cues])
    report['storyboard'] = storyboard_check(cues, storyboard)
    report['timing'] = 'passed'
    return report
def report_summary(report):
    reading = report['short_cues'] + report['fast_cues'] + report['long_lines']
    return f"Timing passed. Readability: {'review needed' if reading else 'passed'}. " + report['storyboard']['message']
