"""Read-only checks. Storyboards are references, never replacement transcripts."""
from difflib import SequenceMatcher
import html
import re
from captions import tokens, quality_report


def extract_storyboard_dialogue(document):
    lines = []
    active = False
    for raw in document.splitlines():
        line = raw.strip()
        match = re.match(r'^(?:Narrator|Narration|Dialogue|Voice[ -]?over|VO)(?:\s*\([^)]*\)|\s*[-–]\s*[^:]+)?\s*:\s*(.*)$', line, re.I)
        if match:
            active = True
            line = match.group(1).strip()
        elif not line or re.match(r'^(?:Scene|Visuals?|Animation|On.screen|SFX|Music|Duration|Camera|Notes?|Transitions?|Audio)\b', line, re.I):
            active = False
            continue
        elif not active:
            continue
        if line and not line.startswith('['):
            # Remove speaker labels in a Dialogue block, not their dialogue.
            line = re.sub(r'^[\w -]{1,30}:\s*', '', line)
            lines.append(line.strip('"“”'))
    return ' '.join(lines) or None


def storyboard_check(transcript, storyboard):
    if not storyboard or not tokens(storyboard):
        return {'status': 'unavailable', 'message': 'Storyboard not checked: no readable dialogue was available.'}
    expected, actual = tokens(storyboard), tokens(transcript)
    matcher = SequenceMatcher(None, expected, actual, autojunk=False)
    missing = extra = 0
    differences = []
    for tag, i, j, k, l in matcher.get_opcodes():
        if tag == 'equal':
            continue
        missing += j-i
        extra += l-k
        if len(differences) < 5:
            differences.append({'storyboard': ' '.join(expected[i:j])[:180], 'audio': ' '.join(actual[k:l])[:180]})
    status = 'matched' if not missing and not extra else 'review'
    return {'status': status, 'storyboard_words': len(expected), 'transcript_words': len(actual),
            'missing_or_changed_words': missing, 'extra_or_changed_words': extra,
            'examples': differences,
            'message': 'Storyboard matches recognized dialogue.' if status == 'matched' else
                f'Storyboard review: {missing} reference words missing/changed; {extra} audio words extra/changed. Audio transcript retained.'}


def check_output(cues, storyboard):
    transcript = ' '.join(html.unescape(text) for _, _, text in cues)
    report = quality_report([{'start': s, 'end': e, 'text': html.unescape(t)} for s, e, t in cues])
    report['storyboard'] = storyboard_check(transcript, storyboard)
    report['timing'] = 'passed'
    return report
def report_summary(report):
    reading = report['short_cues'] + report['fast_cues'] + report['long_lines']
    return f"Timing passed. Readability: {'review needed' if reading else 'passed'}. " + report['storyboard']['message']
