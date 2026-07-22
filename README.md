# hermes-openai-server-compaction

**OpenAI server-side compaction for [Hermes Agent](https://github.com/NousResearch/hermes-agent) — a dual-path context engine plugin.**

When a long Hermes conversation approaches the model's context limit, the
stock behavior is to summarize older turns into a text block with an
auxiliary LLM call. That summary is portable but lossy.

OpenAI's Responses API offers **server-side compaction**: the backend
compacts the conversation itself and returns an opaque, encrypted
`compaction_summary` item that carries full-fidelity conversation state.
This is the same mechanism Codex CLI uses for its own compaction
(`codex-rs/core/src/compact_remote_request.rs`), inspired by
[algal/pi-openai-server-compaction](https://github.com/algal/pi-openai-server-compaction)
which does the equivalent for the Pi agent.

This plugin brings that to Hermes as a **layered dual path**:

| Layer | What | When it's used |
|---|---|---|
| **Portable text summary** | The built-in `ContextCompressor` output, unchanged | Always produced — non-OpenAI models, session exports, model switches |
| **Server compaction artifact** | Encrypted `compaction_summary` item from `POST /responses/compact` | Added on top when the active backend is an OpenAI Responses endpoint |

If the server call fails for any reason, you get exactly stock Hermes
behavior. The plugin never makes compaction less reliable than the built-in
path.

## How it works

1. On compaction, the full transcript is sent to `POST
   {base_url}/responses/compact` using the Codex-CLI-compatible wire shape.
2. The response contains recent messages retained verbatim plus one
   encrypted `compaction_summary` item.
3. The built-in text compression runs as usual (portable layer).
4. The opaque item is attached to the first assistant message of the
   compacted transcript via Hermes's existing `codex_reasoning_items`
   channel — the same channel Hermes already uses to persist and replay
   encrypted reasoning items. Hermes threads it back into every subsequent
   Responses request automatically.
5. Hermes's issuer guard (`_issuer_kind`) drops the artifact automatically
   if you switch to a different provider mid-conversation, preventing
   `invalid_encrypted_content` HTTP 400s. The text summary keeps the
   session coherent in that case.

No Hermes core files are modified. Everything rides public extension
points: the `ContextEngine` ABC, the `register(ctx)` plugin hook, and the
opaque-item replay channel.

## Verified behavior

Tested live against the ChatGPT Codex OAuth backend
(`chatgpt.com/backend-api/codex`, model `gpt-5.6-sol`, Codex CLI wire
shape 0.144.1):

- `/responses/compact` returns HTTP 200 with retained messages + one
  `compaction_summary` item.
- The opaque item survives Hermes's `_chat_messages_to_responses_input`
  adapter untouched and replays cleanly on follow-up `/responses` calls.
- **Strict fidelity check**: facts planted only in assistant turns (which
  get folded into the encrypted blob, absent from all retained plaintext)
  are recalled exactly after compaction — the blob demonstrably carries
  conversation state, not just decoration.

## Requirements

- Hermes Agent with the pluggable context-engine interface
  (`agent/context_engine.py`, mid-2026 or later)
- An OpenAI Responses backend as your main model:
  - ChatGPT Codex OAuth (`openai-codex` provider), or
  - OpenAI platform API key (`api.openai.com`)
- On other providers (xAI, Anthropic, ...) the plugin loads but the server
  path stays dormant — behavior is identical to stock Hermes.

## Install

```bash
git clone https://github.com/zihaofeng2001/hermes-openai-server-compaction \
  ~/.hermes/plugins/openai-server-compaction
```

Enable in `~/.hermes/config.yaml`:

```yaml
plugins:
  enabled:
    - openai-server-compaction

context:
  engine: openai-server-compaction
```

Restart the Hermes gateway (or start a new CLI session). You should see
`Using context engine: openai-server-compaction` in the log at startup.

## Configuration

All optional, under `context.openai_server_compaction`:

```yaml
context:
  engine: openai-server-compaction
  openai_server_compaction:
    enabled: true            # kill switch for the server path only
    reasoning_effort: medium # effort mirrored into the compact request
    request_timeout: 120     # seconds for the unary compact call
```

Compaction thresholds are owned by the engine (same defaults as the
built-in compressor).

## Rollback

Any of these fully disables the plugin:

- Set `context.engine: compressor` (back to built-in), or
- Remove the plugin from `plugins.enabled`, or
- Set `context.openai_server_compaction.enabled: false` to keep the
  engine but disable only the server path.

Existing artifacts in stored sessions are ignored by the issuer guard once
the backend no longer matches; nothing needs cleaning up.

## Data handling — read before enabling

- On compaction, your conversation transcript is sent to OpenAI's
  `/responses/compact` endpoint. This is the same data OpenAI already
  sees on every normal turn of an OpenAI-backed conversation, but be
  aware compaction is one more server-side processing step.
- The returned artifact is encrypted and **not human-readable**; it is
  stored inside your local Hermes session data attached to a marker
  message.
- The artifact is sealed to the issuing endpoint. It cannot be decrypted
  locally or replayed against another provider.

## Tests

Offline unit tests (no Hermes checkout, no network):

```bash
python -m pytest tests/test_plugin.py -v
```

Live end-to-end smoke (requires a Hermes checkout + working OpenAI auth;
makes real API calls):

```bash
python tests/live_smoke.py /path/to/hermes-agent [model]
```

## Repository layout

| File | Purpose |
|---|---|
| `__init__.py` | Plugin registration (`register(ctx)`) |
| `engine.py` | `ServerCompactionEngine` — ContextEngine implementation, dual-path orchestration, issuer stamping |
| `compact_client.py` | Stdlib-only HTTP client for `/responses/compact` (Codex CLI wire shape) |
| `plugin.yaml` | Hermes plugin manifest |
| `tests/test_plugin.py` | Offline unit tests |
| `tests/live_smoke.py` | Live end-to-end verification script |

## Limitations

- **Single-provider artifact.** The encrypted item only helps while you
  stay on the same OpenAI Responses backend. Cross-provider continuity is
  handled by the text-summary layer.
- **The `compaction_summary` passthrough relies on Hermes's
  `codex_reasoning_items` replay channel accepting non-`reasoning` opaque
  items.** This is true of current Hermes (the replay filter keys on
  `encrypted_content`, not `type`) and is covered by the live smoke test,
  but it is not a documented contract. If a future Hermes release
  restricts the channel, the plugin degrades to text-only compression.
- **No multi-artifact chaining strategy beyond "newest wins".** Each new
  compaction folds prior state into a fresh artifact and stale artifacts
  are stripped.
- Recall-quality claims for server-side compaction versus text summaries
  come from upstream projects' benchmarks (see the Pi extension's report);
  this repository verifies the mechanism, not a quality delta.

## License

MIT
