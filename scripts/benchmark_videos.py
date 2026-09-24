"""Run real audio through transcription only; never update delivery folders or CRM.

OPENAI_API_KEY must be configured. Each input makes a paid transcription request;
an opening recheck or validation retry can make additional requests.
"""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('videos', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main.require_openai_api_key()
    args.output.mkdir(parents=True, exist_ok=True)
    collect = main.collect_transcription_data
    for video in args.videos:
        def capture(client, audio, vo_prompt=None):
            words, segments = collect(client, audio, vo_prompt)
            data = {'source': str(video.resolve()), 'duration': len(audio)/1000,
                    'result': {'words': words, 'segments': segments}}
            (args.output / (video.stem + '.json')).write_text(json.dumps(data, indent=2))
            return words, segments
        with patch.object(main, 'collect_transcription_data', side_effect=capture):
            vtt = main.transcribe_file(str(video))
        output = args.output / (video.stem + '.vtt')
        output.write_text(vtt)
        print(output)

if __name__ == '__main__':
    run()
