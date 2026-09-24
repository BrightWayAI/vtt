# VTT generator

FastAPI service for static, phrase-based English WebVTT captions. Requires Python, FFmpeg, the packages in `requirements.txt`, and `OPENAI_API_KEY`.

```sh
python3 -m pip install -r requirements.txt
python3 -m uvicorn main:app --host 127.0.0.1 --port 8000
python3 -m unittest discover -s tests -v
```

Transcription uses `whisper-1` word and segment timestamps. Output has no word highlighting. Word boundaries drive phrase timing; missing/incompatible word metadata uses approximate segment timing while preserving text. Reading duration targets 1.2–6 seconds with up to two 42-character lines. Speech boundaries take priority over minimum duration. Logs report readability exceptions.

`ENABLE_CAPTION_CLEANUP` defaults to `0`. If enabled, cleanup can only change punctuation and case; edits that change words are rejected. Storyboard text is an optional transcription hint, not authoritative replacement dialogue.

The web workflow may write delivered files and update configured Airtable/Sheets integrations. To test audio without those side effects:

```sh
python3 scripts/benchmark_videos.py /path/to/video.mp4 --output /tmp/vtt-benchmark
python3 scripts/audit_captions.py /tmp/vtt-benchmark
```

The first command makes paid OpenAI requests and saves JSON word/segment data and VTT files. The second uses that cache without new requests, checks text preservation and timing, and reports readability metrics. To compare an older renderer, pass `--baseline /path/to/old-main.py` (the file is imported as Python code).

See [the quality audit](audit/QUALITY_AUDIT.md), [final pipeline measurements](audit/pipeline-results.json), and [sample captions](audit/samples/). Transcription agreement is not a substitute for listening review.
