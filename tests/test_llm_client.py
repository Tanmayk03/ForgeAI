"""
Tests for llm_client.generate, with the HTTP layer replaced by a mock
(no Ollama needed). Checks the new json_schema / keep_alive additions and
that Ollama errors are turned into the right error types.
"""

import unittest
from unittest import mock

import requests

from forgeai import config, llm_client


def fake_response(status=200, body=None):
    response = mock.Mock()
    response.status_code = status
    response.json.return_value = body if body is not None else {"response": "ok"}
    response.text = str(body)
    return response


class TestGeneratePayload(unittest.TestCase):
    @mock.patch("forgeai.llm_client.requests.post")
    def test_json_schema_and_keep_alive_are_sent(self, post):
        post.return_value = fake_response(body={"response": '{"a": 1}'})
        llm_client.generate("hi", json_schema={"type": "object"})
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["format"], {"type": "object"})
        self.assertEqual(payload["keep_alive"], config.KEEP_ALIVE)

    @mock.patch("forgeai.llm_client.requests.post")
    def test_without_schema_no_format_is_sent(self, post):
        post.return_value = fake_response()
        self.assertEqual(llm_client.generate("hi"), "ok")
        self.assertNotIn("format", post.call_args.kwargs["json"])


class TestGenerateErrors(unittest.TestCase):
    @mock.patch("forgeai.llm_client.requests.post", side_effect=requests.exceptions.ConnectionError)
    def test_not_running(self, _):
        with self.assertRaises(llm_client.OllamaNotRunningError):
            llm_client.generate("hi")

    @mock.patch("forgeai.llm_client.requests.post", side_effect=requests.exceptions.Timeout)
    def test_timeout(self, _):
        with self.assertRaises(llm_client.LLMTimeoutError):
            llm_client.generate("hi")

    @mock.patch("forgeai.llm_client.requests.post")
    def test_model_not_found(self, post):
        post.return_value = fake_response(404, {"error": "model 'x' not found"})
        with self.assertRaises(llm_client.ModelNotFoundError):
            llm_client.generate("hi")

    @mock.patch("forgeai.llm_client.requests.post")
    def test_server_error_and_empty_reply(self, post):
        post.return_value = fake_response(500, {"error": "out of memory"})
        with self.assertRaises(llm_client.LLMResponseError):
            llm_client.generate("hi")
        post.return_value = fake_response(200, {"response": ""})
        with self.assertRaises(llm_client.LLMResponseError):
            llm_client.generate("hi")


if __name__ == "__main__":
    unittest.main()
