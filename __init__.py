"""OpenAI server-side compaction context engine for Hermes Agent.

This plugin replaces the built-in text-summarization ``ContextCompressor``
with a dual-path compaction engine for OpenAI Responses backends:

1. **Server-side compaction** (primary): calls the OpenAI Responses
   ``/responses/compact`` endpoint. The server returns the recent
   conversation tail as plain messages plus an opaque, encrypted
   ``compaction_summary`` item that carries full-fidelity conversation
   state. That opaque item is attached to a marker message via Hermes's
   existing ``codex_reasoning_items`` replay channel, so it is persisted
   and threaded back into every subsequent Responses request
   automatically.
2. **Portable text summary** (fallback + portability layer): the built-in
   ``ContextCompressor`` summary path. Used when the compact endpoint is
   unavailable, when the active model is not an OpenAI Responses backend,
   or when the server call fails for any reason.

Design constraints honored:
- No Hermes core files are modified. Everything rides public extension
  points: the ``ContextEngine`` ABC, the ``register(ctx)`` plugin hook,
  and the ``codex_reasoning_items`` opaque-item replay channel.
- On non-OpenAI providers (xAI, Anthropic, ...) the engine transparently
  degrades to the built-in text-summarization behavior.
- If a model switch makes an old blob undecryptable, Hermes's issuer
  guard drops it at replay time and the text summary keeps the session
  coherent — the same graceful degradation path Hermes already uses for
  encrypted reasoning items.
"""

try:
    from .engine import ServerCompactionEngine
except ImportError:  # flat (non-package) import — tests, standalone use
    from engine import ServerCompactionEngine

__all__ = ["ServerCompactionEngine", "register"]

ENGINE_NAME = "openai-server-compaction"


def register(ctx):
    """Hermes plugin entry point.

    Called by the Hermes plugin manager when this plugin is enabled via
    ``plugins.enabled`` in config.yaml. Registers the context engine so
    it can be selected with ``context.engine: openai-server-compaction``.
    """
    ctx.register_context_engine(ServerCompactionEngine())
