"""Unit tests for the standalone (non-Hermes) logic of the plugin.

These run WITHOUT a Hermes checkout: they cover the pure helpers in
``engine.py`` and the payload/response handling in ``compact_client.py``.
Integration behavior against a live Hermes install is covered separately
by the live smoke script in ``tests/live_smoke.py``.
"""

import json
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PLUGIN_ROOT))

import compact_client as client_mod  # noqa: E402
import engine as engine_mod  # noqa: E402


class TestBackendDetection(unittest.TestCase):
    def test_chatgpt_codex_backend_is_eligible(self):
        self.assertTrue(engine_mod.is_openai_responses_backend(
            "openai-codex", "codex_responses",
            "https://chatgpt.com/backend-api/codex"))

    def test_platform_api_is_eligible(self):
        self.assertTrue(engine_mod.is_openai_responses_backend(
            "openai", "codex_responses", "https://api.openai.com/v1"))

    def test_xai_responses_is_not_eligible(self):
        self.assertFalse(engine_mod.is_openai_responses_backend(
            "xai-oauth", "codex_responses", "https://api.x.ai/v1"))

    def test_chat_completions_mode_is_not_eligible(self):
        self.assertFalse(engine_mod.is_openai_responses_backend(
            "openai", "chat_completions", "https://api.openai.com/v1"))

    def test_anthropic_is_not_eligible(self):
        self.assertFalse(engine_mod.is_openai_responses_backend(
            "anthropic", "anthropic_messages",
            "https://api.anthropic.com"))


class TestIssuerClassification(unittest.TestCase):
    def test_codex_backend(self):
        self.assertEqual(
            engine_mod.classify_issuer(
                "https://chatgpt.com/backend-api/codex"),
            "codex_backend")

    def test_other_url_is_stamped_verbatim(self):
        self.assertEqual(
            engine_mod.classify_issuer("https://api.openai.com/v1"),
            "other:https://api.openai.com/v1")

    def test_empty(self):
        self.assertEqual(engine_mod.classify_issuer(""), "other")


class TestOpaqueItemHandling(unittest.TestCase):
    def _opaque(self, blob="enc"):
        return {"type": "compaction_summary", "encrypted_content": blob}

    def test_attach_prefers_assistant_message(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "summary marker"},
            {"role": "assistant", "content": "ack"},
        ]
        ok = engine_mod.attach_opaque_items(
            messages, [self._opaque()], "codex_backend")
        self.assertTrue(ok)
        items = messages[2]["codex_reasoning_items"]
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["_issuer_kind"], "codex_backend")
        # original dict not mutated
        self.assertNotIn("_issuer_kind", self._opaque())

    def test_attach_appends_to_existing_reasoning_items(self):
        messages = [{
            "role": "assistant", "content": "a",
            "codex_reasoning_items": [
                {"type": "reasoning", "encrypted_content": "r1"}],
        }]
        engine_mod.attach_opaque_items(
            messages, [self._opaque()], "codex_backend")
        self.assertEqual(len(messages[0]["codex_reasoning_items"]), 2)

    def test_attach_fails_without_assistant_message(self):
        messages = [{"role": "user", "content": "u"}]
        self.assertFalse(engine_mod.attach_opaque_items(
            messages, [self._opaque()], "codex_backend"))

    def test_strip_removes_only_compaction_items(self):
        messages = [{
            "role": "assistant", "content": "a",
            "codex_reasoning_items": [
                {"type": "reasoning", "encrypted_content": "keep"},
                self._opaque("stale"),
            ],
        }]
        removed = engine_mod.strip_stale_opaque_items(messages)
        self.assertEqual(removed, 1)
        kept = messages[0]["codex_reasoning_items"]
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["type"], "reasoning")

    def test_strip_drops_empty_key(self):
        messages = [{
            "role": "assistant", "content": "a",
            "codex_reasoning_items": [self._opaque("stale")],
        }]
        engine_mod.strip_stale_opaque_items(messages)
        self.assertNotIn("codex_reasoning_items", messages[0])


class TestCompactClient(unittest.TestCase):
    def _fake_response(self, payload, status=200):
        resp = mock.MagicMock()
        resp.status = status
        resp.read.return_value = json.dumps(payload).encode()
        resp.__enter__ = lambda s: s
        resp.__exit__ = lambda s, *a: False
        return resp

    def test_splits_retained_and_opaque(self):
        payload = {"output": [
            {"type": "message", "role": "user", "content": []},
            {"type": "compaction_summary", "encrypted_content": "blob"},
        ], "usage": {"input_tokens": 10}}
        with mock.patch.object(client_mod.urllib.request, "urlopen",
                               return_value=self._fake_response(payload)):
            retained, opaque, usage = client_mod.compact_conversation(
                base_url="https://chatgpt.com/backend-api/codex",
                api_key="k", model="m", input_items=[],
                instructions="i")
        self.assertEqual(len(retained), 1)
        self.assertEqual(len(opaque), 1)
        self.assertEqual(usage["input_tokens"], 10)

    def test_missing_opaque_item_raises(self):
        payload = {"output": [
            {"type": "message", "role": "user", "content": []}]}
        with mock.patch.object(client_mod.urllib.request, "urlopen",
                               return_value=self._fake_response(payload)):
            with self.assertRaises(client_mod.CompactionError):
                client_mod.compact_conversation(
                    base_url="https://api.openai.com/v1",
                    api_key="k", model="m", input_items=[],
                    instructions="i")

    def test_http_error_raises_compaction_error(self):
        import urllib.error
        err = urllib.error.HTTPError(
            "u", 404, "nf", {}, None)
        err.read = lambda: b"not found"
        with mock.patch.object(client_mod.urllib.request, "urlopen",
                               side_effect=err):
            with self.assertRaises(client_mod.CompactionError) as ctx:
                client_mod.compact_conversation(
                    base_url="https://api.openai.com/v1",
                    api_key="k", model="m", input_items=[],
                    instructions="i")
        self.assertIn("404", str(ctx.exception))

    def test_chatgpt_backend_headers(self):
        headers = client_mod.build_headers(
            "tok", is_chatgpt_backend=True, account_id="acc",
            session_id="sess")
        self.assertEqual(headers["originator"], "codex_cli_rs")
        self.assertEqual(headers["OpenAI-Beta"], "responses=experimental")
        self.assertEqual(headers["chatgpt-account-id"], "acc")

    def test_platform_api_headers_are_minimal(self):
        headers = client_mod.build_headers("tok", is_chatgpt_backend=False)
        self.assertNotIn("originator", headers)
        self.assertNotIn("chatgpt-account-id", headers)
        self.assertEqual(headers["Authorization"], "Bearer tok")


class TestEngineFallback(unittest.TestCase):
    """compress() must never be less reliable than the inner compressor."""

    def _engine_with_stub_inner(self):
        eng = engine_mod.ServerCompactionEngine.__new__(
            engine_mod.ServerCompactionEngine)
        eng._config = engine_mod._EngineConfig()
        eng._model = "m"
        eng._base_url = "https://chatgpt.com/backend-api/codex"
        eng._api_key = "k"
        eng._provider = "openai-codex"
        eng._api_mode = "codex_responses"
        eng.context_length = 100000
        inner = types.SimpleNamespace()
        inner.compress = mock.MagicMock(
            side_effect=lambda msgs, **kw: [
                {"role": "user", "content": "[summary]"},
                {"role": "assistant", "content": "tail"}])
        eng._inner = inner
        eng._sync_from_inner = lambda: None
        return eng

    def test_server_failure_falls_back_to_text_only(self):
        eng = self._engine_with_stub_inner()
        with mock.patch.object(
                eng, "_request_server_artifact",
                side_effect=client_mod.CompactionError("boom")):
            out = eng.compress([{"role": "user", "content": "x"}])
        self.assertEqual(len(out), 2)
        self.assertNotIn("codex_reasoning_items", out[1])

    def test_server_success_attaches_artifact(self):
        eng = self._engine_with_stub_inner()
        opaque = [{"type": "compaction_summary",
                   "encrypted_content": "blob"}]
        with mock.patch.object(eng, "_request_server_artifact",
                               return_value=opaque):
            out = eng.compress([{"role": "user", "content": "x"}])
        items = out[1]["codex_reasoning_items"]
        self.assertEqual(items[0]["type"], "compaction_summary")
        self.assertEqual(items[0]["_issuer_kind"], "codex_backend")

    def test_non_openai_backend_skips_server_path(self):
        eng = self._engine_with_stub_inner()
        eng._provider = "xai-oauth"
        eng._base_url = "https://api.x.ai/v1"
        with mock.patch.object(eng, "_request_server_artifact") as req:
            eng.compress([{"role": "user", "content": "x"}])
        req.assert_not_called()


if __name__ == "__main__":
    unittest.main()
