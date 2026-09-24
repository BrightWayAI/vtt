# Caption quality audit — September 24, 2026

Implemented locally; not deployed. Tested three complete videos from `~/Desktop/Videos Uploaded/` (355.56 seconds total). Test runs called transcription directly and did not overwrite delivered captions, update Airtable, or append Google Sheets rows.

## Findings and fixes

1. **Animated output was the default.** Every word created another cue, often lasting only 50–300 milliseconds. Each window's final cue could extend to the whole segment's end, forcing following words forward. Removed this renderer entirely. Captions now display complete phrases, with acoustic word boundaries, at most two lines, a 42-character line target, and a six-second duration limit.
2. **The pipeline fabricated opening dialogue.** A missing opening triggered a strongly prompted retry, followed by insertion of a stock sentence at time zero. Removed both. A generic audio-only opening recheck can recover confidently recognized segments before existing speech. It never supplies expected dialogue or inserts a fallback sentence.
3. **Text-only cleanup could change what was said.** Its prompt allowed script-supported edits although the model did not receive audio. Cleanup is off by default. If enabled, a lexical guard rejects additions, deletions, substitutions, and reordering; punctuation and casing remain allowed.
4. **Word metadata could lose or duplicate content.** The old renderer selected overlapping segment windows, and cleanup updates silently failed when token counts differed. Each word now belongs to at most one segment. Incompatible or missing word metadata falls back to all of the segment text, with approximate timing. Short sentence tails are rebalanced using neighboring word timestamps.
5. **Validation did not enforce media bounds and tolerated 100 ms overlaps.** Added finite/range checks and millisecond-level overlap validation. Rendering escapes WebVTT markup. Readability diagnostics count short cues, fast cues (>22 characters/second), and overlong lines.
6. **Audio was lossy and chunks cut arbitrarily.** Normalized mono 16 kHz, 16-bit WAV preserves input speech without another MP3 encoding. Ten-minute chunks fit below the API upload limit; long-file cuts prefer silence in the preceding ten seconds. Each chunk gets its own one-second leading pad, with offsets removed afterward.
7. **API failures leaked temporary audio; async routes blocked the event loop.** Audio files now close and unlink on failure. Upload and URL transcription work runs in a thread pool. Docker includes the new caption module.

## Real-video evidence

First, the same fresh Whisper word/segment results were sent to both renderers to isolate rendering differences. See `real-video-results.json`. Then all three videos ran through the revised `transcribe_file` pipeline with new paid API calls; see `pipeline-results.json` and `samples/`.

| Video | Old renderer cues | Old median duration | Final pipeline cues | Final median duration |
|---|---:|---:|---:|---:|
| The Dust Bowl | 246 | 0.280 s | 29 | 2.765 s |
| Lin-Manuel Miranda | 261 | 0.200 s | 33 | 2.647 s |
| The Panama Canal | 396 | 0.220 s | 42 | 2.912 s |

Final outputs: zero cues under one second, zero overlaps, zero lines above 42 characters, zero cues above 22 characters/second, and no cue beyond the video duration. Panama Canal retains one 1.04-second cue, below the 1.2-second target; preserving the following speech boundary takes priority over extending it.

The same-input rendering comparison preserved every recognized word. This measures preservation, not recognition accuracy.

## Recognition check

An independent `gpt-4o-transcribe` pass processed each complete video's audio. Comparison ignores punctuation/case and normalizes hyphens to word boundaries. This is agreement between recognizers, not a human-certified word-error-rate benchmark.

- The original full-track Whisper result omitted the first ~10 seconds of Miranda, including its opening and two character lines. An unprompted 15-second excerpt recovered them. The final lossless/padded full pipeline captured all those lines and matched the independent transcript lexically.
- Panama Canal's final transcript matched the independent transcript lexically, including William Gorgas and the ending question.
- Dust Bowl's final transcript disagreed on one interjection: Whisper includes “Yay!” after the opening; the independent recognizer omits it. Flag for listening review. Earlier MP3 transcription also differed on “called” versus “call”; the final pipeline and independent pass agree on “call.”

Whisper remains the timestamp source. The independent recognizer is a benchmark only, not a production dependency. OpenAI's [speech-to-text documentation](https://developers.openai.com/api/docs/guides/speech-to-text) describes the model and timestamp options. Simply switching `WHISPER_MODEL` to a text-only transcription model is not compatible with this pipeline's verbose timestamp request.

## Validation and remaining work

- 16 automated tests cover text preservation, phrase timing, silence, tail rebalancing, missing metadata, segment boundaries, markup escaping, cleanup guards, invalid bounds, failed-call cleanup, opening recovery/no fabrication, chunk offsets, timestamp rounding, upload response, and malformed VTT rejection.
- No human listening pass or browser playback review was completed. The sample VTTs are review artifacts, not replacements for delivered files.
- Recheck a broader corpus with accents, noisy music, overlapping speakers, long videos, and manual reference transcripts before claiming general recognition accuracy. Long-file chunk behavior has synthetic regression coverage, not a >10-minute real-video trial.
- Leading-gap recovery is deliberately conservative and may leave missing speech if it cannot recover a non-overlapping audio segment. Internal omissions still need a fuller speech-coverage review.
- Readability diagnostics are logged; they do not yet appear in the UI. Segment-only timing is approximate, and unusually long unbroken tokens may exceed the line target.
- Historical repair scripts (`finalize_vtts_from_storyboards.py`, `repair_large_gaps_from_storyboard.py`, `replace_vtt_tails_with_audio.py`, `delivery_vtt_repair.py`) can replace dialogue from scripts or hardcoded text and allocate timing proportionally. They are separate from the service; do not use them as automated accuracy validation.
- Existing upload routes combine caption generation with delivery-folder writes, thumbnail generation, and Airtable/Sheets updates. Test tools bypass these side effects. A future separation would make production QA safer and retries easier to reason about.
- Other service audit items: unbounded URL download size, whole-upload memory buffering, unrestricted remote URLs, and unescaped filenames in history HTML. These were outside the caption-quality patch.


## Follow-up: transcript-only workflow

Removed thumbnail creation/routes, delivery-folder writes, Airtable updates, Google Sheets/CSV actions, database history writes/routes, and the integration debug endpoint from the running service. Batch completion now carries the VTT itself for browser download.

Added full storyboard comparison (not the truncated ASR prompt), explicit unavailable/matched/review status, UI/header summaries, and detailed discrepancy examples in the UI. Storyboard lookup is read-only and automatic from the filename's video ID. Runtime text-preservation validation supplements timing checks. The downloaded VTT remains caption-only. Automated coverage includes absent/mismatching storyboards, full dialogue extraction, transcript-only processing, automatic filename-ID lookup, removed routes, and database-free batch downloads. JavaScript syntax also checked. Storyboard integration has mocked connector coverage; a live authenticated Google Doc comparison was not run in this follow-up.
