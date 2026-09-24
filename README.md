# VTT generator

FastAPI service for static, phrase-based English WebVTT captions. Requires Python, FFmpeg, the packages in `requirements.txt`, and `OPENAI_API_KEY`.

```sh
python3 -m pip install -r requirements.txt
python3 -m uvicorn main:app --host 127.0.0.1 --port 8000
python3 -m unittest discover -s tests -v
```

Transcription uses `whisper-1` word and segment timestamps. Output has no word highlighting. Word boundaries drive phrase timing; missing/incompatible word metadata uses approximate segment timing while preserving text. Reading duration targets 1.2–6 seconds with up to two 42-character lines. Speech boundaries take priority over minimum duration. Logs report readability exceptions.

`ENABLE_CAPTION_CLEANUP` defaults to `0`. If enabled, cleanup can only change punctuation and case; edits that change words are rejected. Storyboard text is an optional transcription hint, not authoritative replacement dialogue.

The web workflow only returns a VTT file. It does not generate thumbnails, write delivery-folder copies, store database history, or update Airtable/Google Sheets. Temporary media is deleted after processing. Batch downloads work directly in the browser without a database.

Validation runs before delivery:

- Timestamp bounds, ordering, non-overlap, and valid WebVTT structure.
- Recognized words must survive rendering unchanged; readability exceptions are reported.
- Full storyboard dialogue is compared against the transcript. Missing/changed/extra words are flagged for review, without rewriting the audio transcript. A missing or unreadable storyboard is explicitly marked as not checked.

For numbered video filenames, `AIRTABLE_TOKEN` enables a read-only lookup of the linked Google Doc; the document must be readable by the server. A single-video upload or URL request can also provide `storyboard_text` (the UI has a paste field). Batch items independently look up their own storyboard. Validation summaries appear beside downloads and in `X-Validation-Summary`; detailed results travel inside a non-displaying WebVTT `NOTE` block, so no second file is created.

To benchmark audio directly:

```sh
python3 scripts/benchmark_videos.py /path/to/video.mp4 --output /tmp/vtt-benchmark
python3 scripts/audit_captions.py /tmp/vtt-benchmark
```

The first command makes paid OpenAI requests and saves JSON word/segment data and VTT files. The second uses that cache without new requests, checks text preservation and timing, and reports readability metrics. To compare an older renderer, pass `--baseline /path/to/old-main.py` (the file is imported as Python code).

See [the quality audit](audit/QUALITY_AUDIT.md), [final pipeline measurements](audit/pipeline-results.json), and [sample captions](audit/samples/). Transcription agreement is not a substitute for listening review.
