import json
import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import main
from captions import phrase_cues, tokens
from pydub import AudioSegment


def speech(text, start=0., step=.3):
    words=[{'word':t,'start':start+i*step,'end':start+(i+1)*step} for i,t in enumerate(text.split())]
    return words,[{'text':text,'start':start,'end':words[-1]['end']}]


class CaptionsTest(unittest.TestCase):
    def test_long_sentence_preserves_words_without_animation(self):
        text=' '.join(['These words should appear once in readable phrases']*8)+'.'
        words,segs=speech(text)
        vtt=main.build_caption_vtt(segs,words,30)
        cues=main.parse_vtt_cues(vtt)
        self.assertEqual(tokens(' '.join(t for _,_,t in cues)),tokens(text))
        self.assertNotIn('<v>',vtt)
        self.assertTrue(main.validate_vtt_output(vtt,30)[0])
        self.assertLess(len(cues),len(words)/3)
        self.assertTrue(all(len(line)<=42 for line in vtt.splitlines() if '-->' not in line))

    def test_silence_not_filled_or_later_speech_delayed(self):
        w1,s1=speech('Wait here.',0)
        w2,s2=speech('We are back now.',10)
        cues=phrase_cues(w1+w2,s1+s2,15)
        self.assertLess(cues[0]['end'],2)
        self.assertEqual(cues[1]['start'],10)

    def test_segment_text_survives_missing_word_metadata(self):
        words,segs=speech('We saw twenty three birds today.')
        words.pop(2)
        cues=phrase_cues(words,segs,4)
        self.assertEqual(tokens(' '.join(c['text'] for c in cues)),tokens(segs[0]['text']))

    def test_boundary_word_not_duplicated(self):
        words,segs=speech('One two three four five six seven eight.')
        segs=[{'start':0.,'end':1.2,'text':'One two three four'},
              {'start':1.2,'end':2.4,'text':'five six seven eight.'}]
        cues=phrase_cues(words,segs,3)
        self.assertEqual(tokens(' '.join(c['text'] for c in cues)),tokens('One two three four five six seven eight.'))

    def test_rebalances_sentence_tail(self):
        words,segs=speech('Today his musicals are performed around the world and many people watch them at home.',step=.32)
        cues=phrase_cues(words,segs,8)
        self.assertTrue(all(len(c['text'].split())>=3 for c in cues))
        self.assertTrue(all(c['end']-c['start']>=1.2 for c in cues))

    def test_escape_vtt_markup(self):
        vtt=main.build_caption_vtt([{'start':0.,'end':3.,'text':'Use <tag> & keep it.'}],audio_duration_s=3)
        self.assertIn('&lt;tag&gt; &amp;',vtt)

    def test_validation_rejects_bad_bounds_and_overlap(self):
        for seg in [{'start':-1.,'end':2.,'text':'Some words'},
                    {'start':1.,'end':20.,'text':'Some words'},
                    {'start':math.nan,'end':2.,'text':'Some words'}]:
            self.assertFalse(main.validate_segments_for_vtt([seg],5)[0])
        vtt='WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nSome words\n\n00:00:01.950 --> 00:00:03.000\nMore words\n'
        self.assertFalse(main.validate_vtt_output(vtt,5)[0])

    def test_cleanup_cannot_invent_or_delete_words(self):
        batch=[{'start':0.,'end':2.,'text':'The ship is here.'}]
        for replacement in ['', 'The ship has arrived.', "Where did you go this time Riptoes?"]:
            client=Mock()
            client.chat.completions.create.return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps([replacement])))])
            self.assertEqual(main.cleanup_segment_batch(client,batch),batch)

    def test_cleanup_can_fix_case_and_punctuation(self):
        batch=[{'start':0.,'end':2.,'text':'the ship is here'}]
        client=Mock()
        client.chat.completions.create.return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='["The ship is here."]'))])
        self.assertEqual(main.cleanup_segment_batch(client,batch)[0]['text'],'The ship is here.')

    def test_failed_transcription_removes_temp_file(self):
        paths=[]
        def fail(client,path,prompt=None):
            paths.append(path)
            raise RuntimeError('API unavailable')
        with patch.object(main,'transcribe_chunk',side_effect=fail):
            with self.assertRaises(RuntimeError):main.collect_transcription_data(Mock(),AudioSegment.silent(duration=2000))
        from pathlib import Path
        self.assertTrue(paths)
        self.assertTrue(all(not Path(p).exists() for p in paths))

    def test_opening_recheck_does_not_insert_stock_dialogue(self):
        seg=SimpleNamespace(start=7.,end=9.,text='Actual words here.')
        first=SimpleNamespace(words=[],segments=[seg])
        empty=SimpleNamespace(words=[],segments=[])
        with patch.object(main,'transcribe_audio_clip',side_effect=[first,empty]):
            _,segments=main.collect_transcription_data(Mock(),AudioSegment.silent(duration=10000))
        self.assertEqual([s['text'] for s in segments],['Actual words here.'])
        self.assertEqual(segments[0]['start'],6.)

    def test_opening_recovery_uses_audio_segments(self):
        first=SimpleNamespace(words=[],segments=[SimpleNamespace(start=8.,end=10.,text='Main narration.')])
        opening=SimpleNamespace(words=[],segments=[SimpleNamespace(start=2.,end=4.,text='Look at that costume!')])
        with patch.object(main,'transcribe_audio_clip',side_effect=[first,opening]):
            _,segments=main.collect_transcription_data(Mock(),AudioSegment.silent(duration=12000))
        self.assertEqual([s['text'] for s in segments],['Look at that costume!','Main narration.'])
        self.assertEqual(segments[0]['start'],1.)

    def test_chunk_offsets_and_contiguous_coverage(self):
        audio=AudioSegment.silent(duration=5000)
        with patch.object(main,'CHUNK_DURATION_MS',2000):
            ranges=list(main.audio_chunk_ranges(audio))
        self.assertEqual(ranges[0][0],0)
        self.assertEqual(ranges[-1][1],5000)
        self.assertTrue(all(a[1]==b[0] for a,b in zip(ranges,ranges[1:])))
        result=SimpleNamespace(words=[],segments=[SimpleNamespace(start=1.,end=2.,text='A spoken phrase.')])
        with patch.object(main,'audio_chunk_ranges',return_value=iter([(0,2000),(2000,5000)])), patch.object(main,'transcribe_audio_clip',return_value=result):
            _,segments=main.collect_transcription_data(Mock(),audio)
        self.assertEqual([s['start'] for s in segments],[0.,2.])

    def test_upload_returns_static_vtt(self):
        from fastapi.testclient import TestClient
        words,segs=speech('These are stable captions for the video.')
        vtt=main.build_caption_vtt(segs,words,4)
        with patch.object(main,'require_openai_api_key'), patch.object(main,'process_media_file',return_value={'vtt_text':vtt,'validation':'Timing passed.'}) as process:
            response=TestClient(main.app).post('/transcribe',files={'file':('example.mp4',b'test media','video/mp4')})
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.text,vtt)
        self.assertIn('example.vtt',response.headers['content-disposition'])
        from pathlib import Path
        self.assertFalse(Path(process.call_args.args[0]).exists())

    def test_malformed_timestamp_cannot_be_ignored(self):
        vtt='WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nSome valid words\n\nBAD --> BAD\nMissing caption\n'
        self.assertFalse(main.validate_vtt_output(vtt,5)[0])

    def test_timestamp_rounding_carries_hours(self):
        self.assertEqual(main.fmt_ts(3599.9999),'01:00:00.000')

if __name__=='__main__':unittest.main()
