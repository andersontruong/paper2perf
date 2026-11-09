from pathlib import Path
from dataclasses import replace
from email.utils import formatdate
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, patch

import httpx

from paper2perf.config import Settings
from paper2perf.llm import ModelClient, ModelError, OutputLimitError, StructuredOutputError, retry_after_seconds
from paper2perf.schemas import Answer
from paper2perf.store import Store


class ModelClientTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.store = Store(Path(self.temp.name))
        self.sid = self.store.new_session()
        self.rid = self.store.start_run(self.sid, {})
        self.settings = Settings(data_dir=Path(self.temp.name), model="test-model", api_key="test-only-key")
        self.client = ModelClient(self.settings, self.store, self.rid)

    def tearDown(self):
        self.temp.cleanup()

    def response(self, content):
        response = httpx.Response(200, request=httpx.Request("POST", "https://provider.test/chat/completions"),
                                  json={"model": "test-model", "choices": [{"message": {"content": content}}]})
        context = MagicMock()
        context.__enter__.return_value.post.return_value = response
        return context

    def test_records_requests_without_credentials(self):
        with patch("paper2perf.llm.httpx.Client", return_value=self.response('{"claims":[],"limitations":[]}')):
            value = self.client.ask("answer", {"question": "test"}, Answer)
        self.assertEqual(value.claims, [])
        folder = self.store.root / "runs" / self.rid
        self.assertEqual(len(list(folder.glob("llm-*-request.json"))), 1)
        self.assertTrue(list(folder.glob("llm-*-response.json")))
        self.assertNotIn("test-only-key", "".join(p.read_text() for p in folder.iterdir()))

    def test_malformed_contract_is_a_repairable_error(self):
        with patch("paper2perf.llm.httpx.Client", return_value=self.response('{"wrong":1}')):
            with self.assertRaises(StructuredOutputError):
                self.client.ask("answer", {}, Answer)

    def test_model_budget_blocks_calls(self):
        self.client.calls = self.settings.max_model_calls
        with self.assertRaisesRegex(ModelError, "budget"):
            self.client.ask("answer", {}, Answer)

    def raw_response(self, status=200, content='{"claims":[],"limitations":[]}', *, code=None,
                     error_type=None, headers=None, finish="stop", usage=None, message="Provider error"):
        data = {"error": {"code": code, "type": error_type, "message": message}} if status >= 400 else {
            "model": "test-model", "usage": usage,
            "choices": [{"finish_reason": finish, "message": {"content": content}}]
        }
        return httpx.Response(status, json=data, headers=headers or {},
                              request=httpx.Request("POST", "https://provider.test/chat/completions"))

    def scripted_http(self, responses):
        context = MagicMock()
        context.__enter__.return_value.post.side_effect = responses
        return context

    def test_rate_limit_honors_retry_after_then_succeeds(self):
        context = self.scripted_http([
            self.raw_response(429, code="rate_limit_exceeded", headers={"Retry-After": "7", "x-request-id": "req-test"}),
            self.raw_response()
        ])
        with patch("paper2perf.llm.httpx.Client", return_value=context), \
                patch("paper2perf.llm.time.sleep") as sleep, patch("paper2perf.llm.random.uniform", return_value=0):
            result = self.client.ask("answer", {}, Answer)
        self.assertEqual(result.claims, [])
        sleep.assert_called_once_with(7)
        self.assertEqual(self.client.calls, 2)
        diagnostic = (self.store.root / "runs" / self.rid / "model-error.json").read_text()
        self.assertIn("rate_limit_exceeded", diagnostic)
        self.assertIn("req-test", diagnostic)

    def test_quota_and_spend_limits_do_not_retry(self):
        for code in ["insufficient_quota", "organization_spend_limit_exceeded", "project_spend_limit_exceeded"]:
            with self.subTest(code=code):
                context = self.scripted_http([self.raw_response(429, code=code)])
                client = ModelClient(self.settings, self.store, self.rid)
                with patch("paper2perf.llm.httpx.Client", return_value=context), \
                        patch("paper2perf.llm.time.sleep") as sleep:
                    with self.assertRaisesRegex(ModelError, "quota/billing"):
                        client.ask("answer", {}, Answer)
                sleep.assert_not_called()
                self.assertEqual(client.calls, 1)

    def test_retry_backoff_is_bounded(self):
        context = self.scripted_http([self.raw_response(429)] * 4)
        with patch("paper2perf.llm.httpx.Client", return_value=context), \
                patch("paper2perf.llm.time.sleep") as sleep, patch("paper2perf.llm.random.uniform", return_value=0):
            with self.assertRaisesRegex(ModelError, "HTTP 429"):
                self.client.ask("answer", {}, Answer)
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [2, 4, 8])
        self.assertEqual(self.client.calls, 4)

    def test_long_server_delay_is_not_shortened(self):
        context = self.scripted_http([self.raw_response(429, headers={"Retry-After": "120"})])
        with patch("paper2perf.llm.httpx.Client", return_value=context), \
                patch("paper2perf.llm.time.sleep") as sleep:
            with self.assertRaisesRegex(ModelError, "120s"):
                self.client.ask("answer", {}, Answer)
        sleep.assert_not_called()
        self.assertEqual(self.client.calls, 1)

    def test_retry_attempts_count_toward_model_budget(self):
        client = ModelClient(replace(self.settings, max_model_calls=2), self.store, self.rid)
        context = self.scripted_http([self.raw_response(429)] * 2)
        with patch("paper2perf.llm.httpx.Client", return_value=context), patch("paper2perf.llm.time.sleep"):
            with self.assertRaisesRegex(ModelError, "budget exhausted"):
                client.ask("answer", {}, Answer)
        self.assertEqual(client.calls, 2)

    def test_authentication_error_is_redacted_and_not_retried(self):
        context = self.scripted_http([self.raw_response(401,
            message="Invalid key: test-only-key", headers={"Authorization": "Bearer test-only-key"})])
        with patch("paper2perf.llm.httpx.Client", return_value=context), patch("paper2perf.llm.time.sleep") as sleep:
            with self.assertRaises(ModelError) as error:
                self.client.ask("answer", {}, Answer)
        self.assertNotIn("test-only-key", str(error.exception))
        records = "".join(p.read_text() for p in (self.store.root / "runs" / self.rid).iterdir())
        self.assertNotIn("test-only-key", records)
        sleep.assert_not_called()

    def test_empty_length_response_reports_output_limit_without_repeating(self):
        context = self.scripted_http([self.raw_response(content="", finish="length",
            usage={"completion_tokens": 6000, "completion_tokens_details": {"reasoning_tokens": 6000}})])
        with patch("paper2perf.llm.httpx.Client", return_value=context):
            with self.assertRaisesRegex(OutputLimitError, "reasoning tokens"):
                self.client.ask("answer", {}, Answer)
        self.assertEqual(self.client.calls, 1)
        response = (self.store.root / "runs" / self.rid / "llm-01-answer-response.json").read_text()
        self.assertIn('"finish_reason": "length"', response)
        self.assertIn('"reasoning_tokens": 6000', response)

    def test_empty_content_is_not_reported_as_a_json_contract_failure(self):
        context = self.scripted_http([self.raw_response(content="", finish="stop")])
        with patch("paper2perf.llm.httpx.Client", return_value=context):
            with self.assertRaisesRegex(ModelError, "empty assistant content") as error:
                self.client.ask("answer", {}, Answer)
        self.assertNotIsInstance(error.exception, StructuredOutputError)

    def test_configurable_completion_limit_reasoning_and_pacing(self):
        settings = replace(self.settings, model_max_completion_tokens=8000, model_reasoning_effort="low",
                           model_min_interval_seconds=5)
        client = ModelClient(settings, self.store, self.rid)
        context = self.scripted_http([self.raw_response(), self.raw_response()])
        with patch("paper2perf.llm.httpx.Client", return_value=context), \
                patch("paper2perf.llm.time.monotonic", return_value=100), \
                patch("paper2perf.llm.time.sleep") as sleep:
            client.ask("answer", {}, Answer)
            client.ask("answer", {}, Answer)
        sleep.assert_called_once_with(5)
        body = context.__enter__.return_value.post.call_args.kwargs["json"]
        self.assertEqual(body["max_completion_tokens"], 8000)
        self.assertEqual(body["reasoning_effort"], "low")

    def test_openrouter_reasoning_off_uses_nested_enabled_flag(self):
        settings = replace(self.settings, base_url="https://openrouter.ai/api/v1", model_reasoning_effort="none")
        client = ModelClient(settings, self.store, self.rid)
        context = self.scripted_http([self.raw_response()])
        with patch("paper2perf.llm.httpx.Client", return_value=context):
            client.ask("answer", {}, Answer)
        body = context.__enter__.return_value.post.call_args.kwargs["json"]
        self.assertEqual(body["reasoning"], {"enabled": False})
        self.assertNotIn("reasoning_effort", body)
        self.assertNotIn("exclude", body["reasoning"])

    def test_openrouter_explicit_effort_is_nested_and_unset_is_omitted(self):
        for effort in ["low", ""]:
            with self.subTest(effort=effort):
                settings = replace(self.settings, base_url="https://openrouter.ai/api/v1", model_reasoning_effort=effort)
                context = self.scripted_http([self.raw_response()])
                with patch("paper2perf.llm.httpx.Client", return_value=context):
                    ModelClient(settings, self.store, self.rid).ask("answer", {}, Answer)
                body = context.__enter__.return_value.post.call_args.kwargs["json"]
                if effort:
                    self.assertEqual(body["reasoning"], {"effort": effort})
                else:
                    self.assertNotIn("reasoning", body)
                self.assertNotIn("reasoning_effort", body)

    def test_token_limit_diagnostic_distinguishes_reasoning_and_visible_content(self):
        import json
        context = self.scripted_http([self.raw_response(content="", finish="length",
            usage={"completion_tokens": 16000, "completion_tokens_details": {"reasoning_tokens": 16000}})])
        with patch("paper2perf.llm.httpx.Client", return_value=context):
            with self.assertRaises(OutputLimitError) as error:
                self.client.ask("answer", {}, Answer)
        diagnostic = json.loads((self.store.root / "runs" / self.rid / "model-error.json").read_text())
        self.assertEqual(diagnostic["reasoning_tokens"], 16000)
        self.assertEqual(diagnostic["assistant_content_chars"], 0)
        self.assertIn("reasoning tokens=16000", str(error.exception))

    def test_missing_reasoning_usage_remains_unknown(self):
        import json
        context = self.scripted_http([self.raw_response(content='{"claims": [', finish="length", usage=None)])
        with patch("paper2perf.llm.httpx.Client", return_value=context):
            with self.assertRaises(OutputLimitError):
                self.client.ask("answer", {}, Answer)
        diagnostic = json.loads((self.store.root / "runs" / self.rid / "model-error.json").read_text())
        self.assertIsNone(diagnostic["reasoning_tokens"])
        self.assertGreater(diagnostic["assistant_content_chars"], 0)

    def test_retry_after_http_date_and_milliseconds(self):
        self.assertEqual(retry_after_seconds(httpx.Headers({"Retry-After": formatdate(107, usegmt=True)}), now=100), 7)
        self.assertEqual(retry_after_seconds(httpx.Headers({"retry-after-ms": "250"})), .25)
        self.assertIsNone(retry_after_seconds({"retry-after": "not-a-date"}))
        self.assertIsNone(retry_after_seconds({"retry-after": "NaN"}))


if __name__ == "__main__":
    unittest.main()
