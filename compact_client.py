"""HTTP client for the OpenAI Responses server-side compaction endpoint.

Standalone module (no Hermes imports) so it can be unit-tested outside a
Hermes checkout. Uses only the Python standard library.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
import uuid
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

COMPACT_PATH = "/responses/compact"

# Wire shape mirrors Codex CLI's remote compaction request
# (codex-rs/core/src/client.rs::compact_conversation_history).
CHATGPT_BACKEND_MARKER = "chatgpt.com"


class CompactionError(Exception):
    """Raised when the compact endpoint returns an error or bad payload."""


def build_headers(
    api_key: str,
    *,
    is_chatgpt_backend: bool,
    account_id: Optional[str] = None,
    session_id: Optional[str] = None,
    client_version: str = "0.144.1",
) -> Dict[str, str]:
    """Build request headers.

    The ChatGPT Codex backend requires the Codex-CLI-compatible header set
    (``originator``, ``OpenAI-Beta``, ``chatgpt-account-id``). The plain
    platform API (``api.openai.com``) only needs the bearer token, but the
    extra headers are harmless there.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if is_chatgpt_backend:
        headers["originator"] = "codex_cli_rs"
        headers["OpenAI-Beta"] = "responses=experimental"
        headers["version"] = client_version
        headers["session_id"] = session_id or str(uuid.uuid4())
        if account_id:
            headers["chatgpt-account-id"] = account_id
    return headers


def compact_conversation(
    *,
    base_url: str,
    api_key: str,
    model: str,
    input_items: List[Dict[str, Any]],
    instructions: str,
    reasoning_effort: str = "medium",
    account_id: Optional[str] = None,
    session_id: Optional[str] = None,
    timeout: float = 120.0,
    prompt_cache_key: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Call ``POST {base_url}/responses/compact`` and split the result.

    Returns ``(retained_message_items, opaque_items, usage)`` where
    ``retained_message_items`` are plain Responses ``message`` items the
    server chose to keep verbatim and ``opaque_items`` are
    ``compaction_summary`` items carrying ``encrypted_content``.

    Raises CompactionError on HTTP or payload-shape failures.
    """
    base = base_url.rstrip("/")
    url = base + COMPACT_PATH
    payload: Dict[str, Any] = {
        "model": model,
        "input": input_items,
        "instructions": instructions,
        "tools": [],
        "parallel_tool_calls": False,
        "reasoning": {"effort": reasoning_effort, "summary": "auto"},
    }
    if prompt_cache_key:
        payload["prompt_cache_key"] = prompt_cache_key

    headers = build_headers(
        api_key,
        is_chatgpt_backend=CHATGPT_BACKEND_MARKER in base,
        account_id=account_id,
        session_id=session_id,
    )
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers=headers, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            status = resp.status
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
        except Exception:  # noqa: BLE001
            pass
        raise CompactionError(
            f"compact endpoint returned HTTP {exc.code}: {detail}"
        ) from exc
    except Exception as exc:  # noqa: BLE001 — network/timeout/etc.
        raise CompactionError(f"compact request failed: {exc}") from exc

    if status != 200:
        raise CompactionError(f"compact endpoint returned HTTP {status}")

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise CompactionError("compact endpoint returned non-JSON body") from exc

    output = _extract_output_items(parsed)
    if output is None:
        raise CompactionError(
            "compact response had no recognizable output item list "
            f"(top-level keys: {sorted(parsed) if isinstance(parsed, dict) else type(parsed).__name__})"
        )

    retained = [it for it in output
                if isinstance(it, dict) and it.get("type") == "message"]
    opaque = [it for it in output
              if isinstance(it, dict)
              and it.get("type") == "compaction_summary"
              and it.get("encrypted_content")]
    if not opaque:
        raise CompactionError(
            "compact response contained no compaction_summary item"
        )
    usage = parsed.get("usage") if isinstance(parsed, dict) else {}
    return retained, opaque, usage if isinstance(usage, dict) else {}


def _extract_output_items(parsed: Any) -> Optional[List[Dict[str, Any]]]:
    """Locate the compacted-history item list in the response payload.

    Observed live shape (ChatGPT Codex backend, 2026-07): a Responses-like
    object with the item list under ``output``. Tolerate a few alternative
    keys and a bare top-level list for forward compatibility.
    """
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        for key in ("output", "items", "input", "history", "data"):
            value = parsed.get(key)
            if isinstance(value, list):
                return value
    return None
