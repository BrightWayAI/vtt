"""Compare cached verbose ASR results without another API call or delivery writes.
Usage: python scripts/audit_captions.py /tmp/vtt-audit --baseline /tmp/vtt_baseline_main.py
"""
import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main
from captions import phrase_cues, quality_report, tokens


def metrics(vtt):
    cues = main.parse_vtt_cues(vtt)
    durations = [e-s for s,e,_ in cues]
    return {'cues':len(cues), 'median_seconds':round(statistics.median(durations),3),
            'under_1_second':sum(d < 1 for d in durations),
            'overlaps':sum(a[1] > b[0]+.001 for a,b in zip(cues,cues[1:])),
            'end':cues[-1][1]}


def run():
    parser=argparse.ArgumentParser();parser.add_argument('cache',type=Path);parser.add_argument('--baseline',type=Path)
    args=parser.parse_args()
    baseline=None
    if args.baseline:
        spec=importlib.util.spec_from_file_location('baseline',args.baseline)
        baseline=importlib.util.module_from_spec(spec);spec.loader.exec_module(baseline)
    results=[]
    for path in sorted(args.cache.glob('*.json')):
        data=json.loads(path.read_text())
        if 'result' not in data: continue
        words,segments=data['result']['words'],data['result']['segments']
        vtt=main.build_caption_vtt(segments,words,data['duration'])
        valid,message=main.validate_vtt_output(vtt,data['duration'])
        cues=phrase_cues(words,segments,data['duration'])
        preserved=tokens(' '.join(s['text'] for s in segments))==tokens(' '.join(c['text'] for c in cues))
        row={'video':Path(data['source']).name,'duration':data['duration'],'new':metrics(vtt),
             'valid':valid,'validation':message,'text_preserved':preserved,'readability':quality_report(cues)}
        if baseline: row['old']=metrics(baseline.build_highlight_vtt(words,segments))
        path.with_suffix('.vtt').write_text(vtt)
        results.append(row)
    print(json.dumps(results,indent=2))
    if any(not r['valid'] or not r['text_preserved'] for r in results): raise SystemExit(1)

if __name__=='__main__':run()
