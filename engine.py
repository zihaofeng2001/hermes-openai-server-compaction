"""ServerCompactionEngine — dual-path context engine for Hermes Agent.

Layering model
--------------
The built-in ``ContextCompressor`` text-summary path is ALWAYS the base
layer: it runs first and its output is what non-OpenAI models, session
exports, and model switches rely on. When the active backend is an OpenAI
Responses endpoint, this engine additionally requests a server-side
compaction artifact (an encrypted ``compaction_summary`` item) and attaches
it to the compacted transcript through Hermes's ``codex_reasoning_items``
replay channel. On every later turn Hermes threads the opaque item back to
the server, restoring full-fidelity conversation state on top of the
portable text summary.

If the server call fails for any reason, the result is exactly the built-in
compression output — this engine never makes compaction less reliable than
stock Hermes.

All Hermes imports are lazy so this module can be imported (and unit
tested) outside a Hermes checkout.
"""

from __future__ import annotations

import copy
import logging
import os
import uuid
from typing import Any, Dict, List, Optional

try:
    from .compact_client import CompactionError, compact_conversation
except ImportError:  # flat (non-package) import — tests, standalone use
    from compact_client import CompactionError, compact_conversation

logger = logging.getLogger(__name__)

OPAQUE_ITEM_TYPE = "compaction_summary"
ISSUER_KIND_KEY = "_issuer_kind"

# Providers that run api_mode=codex_responses but are NOT OpenAI backends.
# xAI's OAuth surface speaks the Responses protocol yet has no
# /responses/compact endpoint, and its encrypted blobs are sealed to xAI.
_NON_OPENAI_RESPONSES_PROVIDERS = {"xai-oauth", "xai"}
_OPENAI_HOST_MARKERS = ("chatgpt.com", "api.openai.com")


# ---------------------------------------------------------------------------
# Pure helpers (no Hermes imports — unit-testable standalone)
# ---------------------------------------------------------------------------

def is_openai_responses_backend(
    provider: str, api_mode: str, base_url: str
) -> bool:
    """True when the active backend supports /responses/compact."""
    if api_mode != "codex_responses":
        return False
    if (provider or "").lower() in _NON_OPENAI_RESPONSES_PROVIDERS:
        return False
    lowered = (base_url or "").lower()
    if any(marker in lowered for marker in _OPENAI_HOST_MARKERS):
        return True
    return (provider or "").lower() in {"openai-codex", "openai"}


def classify_issuer(base_url: str) -> str:
    """Stamp value for the opaque item, matching Hermes's issuer guard.

    Hermes drops replayed opaque items whose ``_issuer_kind`` differs from
    the endpoint currently in use, which prevents HTTP 400
    ``invalid_encrypted_content`` after a model/provider switch. Mirror
    ``agent.codex_responses_adapter._classify_responses_issuer`` for the
    endpoints this engine supports.
    """
    lowered = (base_url or "").lower()
    if "chatgpt.com" in lowered and "/backend-api/codex" in lowered:
        return "codex_backend"
    if base_url:
        return f"other:{base_url}"
    return "other"


def strip_stale_opaque_items(messages: List[Dict[str, Any]]) -> int:
    """Remove compaction_summary items left on surviving messages.

    Each new server compaction supersedes previous artifacts (the compact
    request itself already folded the old blob in). Leaving stale blobs on
    tail messages would replay multiple overlapping artifacts.

    Returns the number of items removed.
    """
    removed = 0
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        items = msg.get("codex_reasoning_items")
        if not isinstance(items, list):
            continue
        kept = [it for it in items
                if not (isinstance(it, dict)
                        and it.get("type") == OPAQUE_ITEM_TYPE)]
        removed += len(items) - len(kept)
        if kept:
            msg["codex_reasoning_items"] = kept
        elif "codex_reasoning_items" in msg:
            del msg["codex_reasoning_items"]
    return removed


def attach_opaque_items(
    messages: List[Dict[str, Any]],
    opaque_items: List[Dict[str, Any]],
    issuer_kind: str,
) -> bool:
    """Attach opaque items to the first assistant message in the tail.

    Hermes only replays ``codex_reasoning_items`` from assistant-role
    messages, so the artifact must ride an assistant message. Attaching to
    an existing message (rather than inserting a new one) preserves the
    strict role-alternation invariant of the transcript.

    Returns True when a carrier message was found and items were attached.
    """
    stamped = [dict(it, **{ISSUER_KIND_KEY: issuer_kind})
               for it in opaque_items]
    # Prefer the earliest assistant message AFTER the summary marker (the
    # summary is role=user in Hermes); fall back to any assistant message.
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            existing = msg.get("codex_reasoning_items")
            if isinstance(existing, list):
                existing.extend(stamped)
            else:
                msg["codex_reasoning_items"] = stamped
            return True
    return False


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class _EngineConfig:
    """Plugin settings, read from ``context.openai_server_compaction``."""

    def __init__(self) -> None:
        self.enabled: bool = True
        self.reasoning_effort: str = "medium"
        self.request_timeout: float = 120.0

    @classmethod
    def load(cls) -> "_EngineConfig":
        cfg = cls()
        try:
            import yaml  # noqa: PLC0415
            home = os.environ.get(
                "HERMES_HOME", os.path.expanduser("~/.hermes"))
            path = os.path.join(home, "config.yaml")
            with open(path, encoding="utf-8") as fh:
                raw = yaml.safe_load(fh) or {}
            section = ((raw.get("context") or {})
                       .get("openai_server_compaction") or {})
            if isinstance(section, dict):
                cfg.enabled = bool(section.get("enabled", cfg.enabled))
                effort = section.get("reasoning_effort")
                if isinstance(effort, str) and effort:
                    cfg.reasoning_effort = effort
                timeout = section.get("request_timeout")
                if isinstance(timeout, (int, float)) and timeout > 0:
                    cfg.request_timeout = float(timeout)
        except Exception:  # noqa: BLE001 — missing/unreadable config is fine
            pass
        return cfg


def _make_base_class():
    """Resolve the ContextEngine ABC lazily.

    Inside Hermes the real ABC is importable; standalone (tests, linting)
    a minimal stub keeps the module importable.
    """
    try:
        from agent.context_engine import ContextEngine  # noqa: PLC0415
        return ContextEngine
    except ImportError:
        class _Stub:  # pragma: no cover — exercised only outside Hermes
            name = "stub"
        return _Stub


class ServerCompactionEngine(_make_base_class()):
    """Dual-path compaction: built-in text summary + OpenAI server artifact."""

    def __init__(self) -> None:
        self._inner = None          # built-in ContextCompressor, built lazily
        self._config = _EngineConfig.load()
        self._session_header_id = str(uuid.uuid4())
        # Model routing state, filled by update_model()
        self._model = ""
        self._base_url = ""
        self._api_key = ""
        self._provider = ""
        self._api_mode = ""
        # ContextEngine surface read directly by the Hermes host
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.threshold_tokens = 0
        self.context_length = 0
        self.compression_count = 0
        self.threshold_percent = 0.75
        self.protect_first_n = 3
        self.protect_last_n = 6

    # -- identity ----------------------------------------------------------

    @property
    def name(self) -> str:
        return "openai-server-compaction"

    def is_available(self) -> bool:
        return True

    # -- inner compressor management --------------------------------------

    def _build_inner(self) -> None:
        """(Re)build the built-in ContextCompressor for the current model."""
        from agent.context_compressor import ContextCompressor  # noqa: PLC0415
        self._inner = ContextCompressor(
            model=self._model,
            base_url=self._base_url,
            api_key=self._api_key,
            provider=self._provider,
            api_mode=self._api_mode,
            quiet_mode=True,
        )
        if self.context_length:
            self._inner.update_model(
                model=self._model,
                context_length=self.context_length,
                base_url=self._base_url,
                api_key=self._api_key,
                provider=self._provider,
                api_mode=self._api_mode,
            )
        self._sync_from_inner()

    def _sync_from_inner(self) -> None:
        inner = self._inner
        if inner is None:
            return
        for attr in ("last_prompt_tokens", "last_completion_tokens",
                     "last_total_tokens", "threshold_tokens",
                     "context_length", "compression_count",
                     "threshold_percent", "protect_first_n",
                     "protect_last_n"):
            value = getattr(inner, attr, None)
            if value is not None:
                setattr(self, attr, value)

    def __deepcopy__(self, memo):
        """Fresh engine per agent; re-derive state via update_model().

        Hermes deep-copies the shared plugin singleton for each child agent.
        The inner compressor may hold session-DB bindings that must not be
        shared, so rebuild instead of copying.
        """
        clone = ServerCompactionEngine()
        memo[id(self)] = clone
        clone._config = copy.copy(self._config)
        if self._model:
            clone.update_model(
                model=self._model,
                context_length=self.context_length,
                base_url=self._base_url,
                api_key=self._api_key,
                provider=self._provider,
                api_mode=self._api_mode,
            )
        return clone

    # -- ContextEngine interface -------------------------------------------

    def update_model(self, model: str, context_length: int,
                     base_url: str = "", api_key: str = "",
                     provider: str = "", api_mode: str = "") -> None:
        self._model = model
        self._base_url = base_url or ""
        self._api_key = api_key or ""
        self._provider = provider or ""
        self._api_mode = api_mode or ""
        self.context_length = context_length
        self._build_inner()

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        if self._inner is not None:
            self._inner.update_from_response(usage)
            self._sync_from_inner()

    def should_compress(self, prompt_tokens: int = None) -> bool:
        if self._inner is None:
            return False
        return self._inner.should_compress(prompt_tokens)

    def should_compress_preflight(self, messages) -> bool:
        if self._inner is None:
            return False
        return self._inner.should_compress_preflight(messages)

    def should_defer_preflight_to_real_usage(self, rough_tokens: int) -> bool:
        if self._inner is None:
            return False
        return self._inner.should_defer_preflight_to_real_usage(rough_tokens)

    def has_content_to_compress(self, messages) -> bool:
        if self._inner is None:
            return True
        return self._inner.has_content_to_compress(messages)

    def bind_session_state(self, **kwargs) -> None:
        binder = getattr(self._inner, "bind_session_state", None)
        if callable(binder):
            binder(**kwargs)

    def on_session_reset(self) -> None:
        super().on_session_reset()
        if self._inner is not None:
            reset = getattr(self._inner, "on_session_reset", None)
            if callable(reset):
                reset()
            self._sync_from_inner()

    # -- core: compress ----------------------------------------------------

    def compress(self, messages: List[Dict[str, Any]],
                 current_tokens: int = None, focus_topic: str = None,
                 **kwargs) -> List[Dict[str, Any]]:
        """Run built-in compression, then layer the server artifact on top."""
        if self._inner is None:
            return messages

        # 1. Server-side compaction on the FULL pre-compression transcript.
        #    Started before the text path so the artifact reflects the same
        #    input; failures are non-fatal by design.
        opaque_items: List[Dict[str, Any]] = []
        if self._server_path_eligible():
            try:
                opaque_items = self._request_server_artifact(messages)
            except CompactionError as exc:
                logger.warning(
                    "Server-side compaction unavailable, falling back to "
                    "text-only compression: %s", exc)
            except Exception as exc:  # noqa: BLE001 — never break compression
                logger.warning(
                    "Unexpected server-compaction failure (text-only "
                    "fallback): %s", exc)

        # 2. Built-in portable text compression (unchanged stock behavior).
        compacted = self._inner.compress(
            messages, current_tokens=current_tokens,
            focus_topic=focus_topic, **kwargs)
        self._sync_from_inner()

        # 3. Attach the artifact to the compacted transcript.
        if opaque_items and compacted is not messages:
            strip_stale_opaque_items(compacted)
            issuer = classify_issuer(self._base_url)
            if attach_opaque_items(compacted, opaque_items, issuer):
                logger.info(
                    "Attached %d server compaction artifact(s) "
                    "(issuer=%s) to compacted transcript",
                    len(opaque_items), issuer)
            else:
                logger.warning(
                    "No assistant message available to carry the server "
                    "compaction artifact; text summary only")
        return compacted

    # -- server path internals ---------------------------------------------

    def _server_path_eligible(self) -> bool:
        return (self._config.enabled
                and bool(self._api_key)
                and is_openai_responses_backend(
                    self._provider, self._api_mode, self._base_url))

    def _request_server_artifact(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """POST the transcript to /responses/compact; return opaque items."""
        from agent.codex_responses_adapter import (  # noqa: PLC0415
            _chat_messages_to_responses_input,
        )
        instructions = ""
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "system":
                content = msg.get("content", "")
                instructions = content if isinstance(content, str) else ""
                break
        input_items = _chat_messages_to_responses_input(
            messages,
            replay_encrypted_reasoning=True,
            current_issuer_kind=classify_issuer(self._base_url),
        )
        retained, opaque, usage = compact_conversation(
            base_url=self._base_url,
            api_key=self._api_key,
            model=self._model,
            input_items=input_items,
            instructions=instructions,
            reasoning_effort=self._config.reasoning_effort,
            account_id=self._resolve_account_id(),
            session_id=self._session_header_id,
            timeout=self._config.request_timeout,
        )
        logger.info(
            "Server compaction ok: %d retained message(s), %d opaque "
            "item(s), usage=%s", len(retained), len(opaque),
            {k: usage.get(k) for k in ("input_tokens", "output_tokens")
             if isinstance(usage, dict)})
        return opaque

    def _resolve_account_id(self) -> Optional[str]:
        """ChatGPT backend needs the account-id header; read it from the
        Hermes auth store when running against that backend."""
        if "chatgpt.com" not in (self._base_url or "").lower():
            return None
        try:
            from hermes_cli.auth import _read_codex_tokens  # noqa: PLC0415
            tokens = _read_codex_tokens().get("tokens") or {}
            account_id = tokens.get("account_id")
            return account_id if isinstance(account_id, str) else None
        except Exception:  # noqa: BLE001
            return None
