import io
import json
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError
import harness

class Response:
    def __init__(self,body):self.body=body
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def read(self):return self.body

class ModelTransportTests(unittest.TestCase):
    def test_transient_gateway_html_retries_identical_request_then_returns_real_model(self):
        error=HTTPError('http://test/model',503,'gateway',{},io.BytesIO(b'<!DOCTYPE html>gateway'))
        calls=[]
        def send(request,timeout):
            calls.append(request)
            if len(calls)==1:raise error
            return Response(json.dumps({'model':'actual-model','choices':[]}).encode())
        with patch.object(harness.urllib.request,'urlopen',side_effect=send),patch.object(harness.time,'sleep') as sleep:
            result=harness.post('http://test/model',{'messages':[]},'')
        self.assertEqual(result['model'],'actual-model')
        self.assertEqual(result['_transport_attempts'],2)
        self.assertIs(calls[0],calls[1])
        self.assertEqual(result['_transport_failures'][0]['status'],503)
        sleep.assert_called_once_with(1.5)

    def test_json_authentication_failure_stops_without_retry_or_fallback(self):
        error=HTTPError('http://test/model',403,'auth',{},io.BytesIO(b'{"error":"please auth first"}'))
        with patch.object(harness.urllib.request,'urlopen',side_effect=error) as send,patch.object(harness.time,'sleep') as sleep:
            result=harness.post('http://test/model',{},'')
        self.assertEqual(send.call_count,1)
        self.assertEqual(result['_status'],403)
        self.assertEqual(result['_transport_attempts'],1)
        sleep.assert_not_called()

    def test_html_authentication_failure_does_not_retry(self):
        for status in (401, 403):
            with self.subTest(status=status):
                error=HTTPError('http://test/model',status,'auth',{},io.BytesIO(b'<html>Access denied</html>'))
                with patch.object(harness.urllib.request,'urlopen',side_effect=error) as send,patch.object(harness.time,'sleep') as sleep:
                    result=harness.post('http://test/model',{},'')
                self.assertEqual(send.call_count,1)
                self.assertEqual(result['_status'],status)
                sleep.assert_not_called()

    def test_network_retries_are_bounded_and_remain_an_error(self):
        with patch.object(harness.urllib.request,'urlopen',side_effect=URLError('unavailable')) as send,patch.object(harness.time,'sleep'):
            result=harness.post('http://test/model',{},'')
        self.assertEqual(send.call_count,3)
        self.assertIn('unavailable',result['_error'])
        self.assertEqual(result['_transport_attempts'],3)
        self.assertNotIn('choices',result)

    def test_invalid_success_response_is_retried_and_never_treated_as_a_decision(self):
        with patch.object(harness.urllib.request,'urlopen',return_value=Response(b'<html>proxy</html>')) as send,patch.object(harness.time,'sleep'):
            result=harness.post('http://test/model',{},'')
        self.assertEqual(send.call_count,3)
        self.assertTrue(result['_error'])


class GenerationSettingsTests(unittest.TestCase):
    def test_pack_override_can_enable_thinking_and_expand_the_response_budget(self):
        with patch.dict(harness.os.environ,{'JEVGAMER_CHAT_THINKING':'0','JEVGAMER_MAX_OUTPUT_TOKENS':'2048'}),patch.object(harness,'post',return_value={}) as post:
            harness.chat([], '', thinking=True,max_output_tokens=4096)
        payload=post.call_args.args[1]
        self.assertTrue(payload['chat_template_kwargs']['thinking'])
        self.assertEqual(payload['max_tokens'],4096)

    def test_explicit_false_overrides_an_enabled_environment_default(self):
        with patch.dict(harness.os.environ,{'JEVGAMER_CHAT_THINKING':'1'}):
            self.assertFalse(harness.chat_generation(thinking=False)['chat_template_kwargs']['thinking'])

    def test_explicit_enabled_environment_mode_is_sent_to_the_service(self):
        with patch.dict(harness.os.environ,{'JEVGAMER_CHAT_THINKING':'1'}):
            self.assertTrue(harness.chat_generation()['chat_template_kwargs']['thinking'])
