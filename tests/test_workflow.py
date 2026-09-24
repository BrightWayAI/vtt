import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
import main
from validation import extract_storyboard_dialogue, storyboard_check

VTT = 'WEBVTT\n\n00:00:00.000 --> 00:00:03.000\nThe ship enters the canal.\n'

class WorkflowTests(unittest.TestCase):
    def test_full_storyboard_not_truncated_and_directions_excluded(self):
        document = 'Narrator: '+ 'A spoken sentence. '*100 + '\nVisual: A ship\nDialogue:\nSam: Look at that!\nScene 2\nCamera: Pan left'
        text = extract_storyboard_dialogue(document)
        self.assertGreater(len(text),900)
        self.assertIn('Look at that!',text)
        self.assertNotIn('Pan left',text)
        self.assertNotIn('A ship',text)

    def test_storyboard_mismatch_is_review_not_replacement(self):
        with patch.object(main,'transcribe_file',return_value=VTT), \
             patch.object(main,'fetch_airtable_record',return_value=('record','Topic','url')), \
             patch.object(main,'fetch_storyboard_vo',return_value='The train enters the station.'):
            result=main.process_media_file('unused','107_video.mp4')
        self.assertIn('The ship enters the canal.',result['vtt_text'])
        self.assertIn('Storyboard review',result['validation'])
        self.assertNotIn('NOTE Validation',result['vtt_text'])
        self.assertTrue(main.validate_vtt_output(result['vtt_text'],3)[0])

    def test_absent_storyboard_is_not_a_pass(self):
        self.assertEqual(storyboard_check('Some words',None)['status'],'unavailable')

    def test_exact_storyboard_match(self):
        self.assertEqual(storyboard_check('The ship enters the canal!','the ship enters the canal.')['status'],'matched')

    def test_processing_has_no_delivery_side_effects(self):
        with patch.object(main,'transcribe_file',return_value=VTT), patch.object(main,'fetch_airtable_record',return_value=('record','Topic','url')), patch.object(main,'fetch_storyboard_vo',return_value='The ship enters the canal.'), patch('pathlib.Path.write_text',side_effect=AssertionError('Unexpected file write')), patch('httpx.Client.post',side_effect=AssertionError('Unexpected POST')), patch('httpx.Client.patch',side_effect=AssertionError('Unexpected PATCH')):
            result=main.process_media_file('unused','107_video.mp4')
        self.assertIn('Storyboard matches',result['validation'])
        for forbidden in ['generate_thumbnail','save_to_upload_folder','update_upload_date','write_to_upload_sheet','get_db','save_transcription']:
            self.assertFalse(hasattr(main,forbidden))

    def test_upload_uses_automatic_storyboard_and_returns_validation(self):
        with patch.object(main,'require_openai_api_key'), patch.object(main,'transcribe_file',return_value=VTT), \
             patch.object(main,'fetch_airtable_record',return_value=('record','Topic','url')), \
             patch.object(main,'fetch_storyboard_vo',return_value='The ship enters the canal.'):
            response=TestClient(main.app).post('/transcribe',files={'file':('107_test.mp4',b'media')})
        self.assertEqual(response.status_code,200)
        self.assertIn('Storyboard matches',response.headers['X-Validation-Summary'])
        self.assertNotIn('X-Thumbnail-URL',response.headers)

    def test_upload_automatically_uses_filename_id_for_storyboard(self):
        with patch.object(main,'require_openai_api_key'), patch.object(main,'transcribe_file',return_value=VTT), \
             patch.object(main,'fetch_airtable_record',return_value=('record','Topic','url')) as lookup, \
             patch.object(main,'fetch_storyboard_vo',return_value='The ship enters the canal.'):
            response=TestClient(main.app).post('/transcribe',files={'file':('107_test.mp4',b'media')})
        lookup.assert_called_once_with(107)
        self.assertIn('Storyboard matches',response.headers['X-Validation-Summary'])

    def test_ui_supports_multi_file_upload_and_keeps_vtt_clean(self):
        self.assertIn('id="file-input" multiple', main.HTML_PAGE)
        self.assertNotIn('id="storyboard"', main.HTML_PAGE)
        with patch.object(main,'transcribe_file',return_value=VTT), \
             patch.object(main,'fetch_airtable_record',return_value=None):
            result=main.process_media_file('unused','107_video.mp4')
        self.assertNotIn('NOTE Validation', result['vtt_text'])
        self.assertTrue(result['vtt_text'].startswith('WEBVTT'))

    def test_batch_returns_downloadable_vtt_without_database(self):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            path=tmp.name
        with patch.object(main,'require_openai_api_key'), patch.object(main,'download_url_to_temp',return_value=(path,'video.mp4',1)), patch.object(main,'transcribe_file',return_value=VTT):
            response=TestClient(main.app).get('/batch',params={'urls':json.dumps(['https://example.com/video.mp4'])})
        messages=[json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')]
        result=next(m for m in messages if m['status']=='done')
        self.assertIn('WEBVTT',result['vtt_text'])
        self.assertIn('not checked',result['validation'])
        self.assertFalse(Path(path).exists())

    def test_removed_routes_not_accessible(self):
        client=TestClient(main.app)
        for path in ['/thumbnail/test','/history','/debug','/download/1']:
            self.assertEqual(client.get(path).status_code,404)

if __name__=='__main__':unittest.main()
