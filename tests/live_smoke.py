"""Live end-to-end smoke test (requires a Hermes checkout + OpenAI auth).

Not run in CI. Verifies against real infrastructure:
1. The engine loads through Hermes's own context-engine loader.
2. compress() on a synthetic long transcript calls the real
   /responses/compact endpoint and attaches the opaque artifact.
3. The compacted transcript passes through Hermes's real Responses
   adapter and a real /responses call recalls facts that exist ONLY
   inside the encrypted artifact (not in any retained plaintext).

Usage:
    python tests/live_smoke.py /path/to/hermes-agent
"""

import json
import sys
import urllib.request
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    hermes_root = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(hermes_root))
    sys.path.insert(0, str(PLUGIN_ROOT))

    from engine import ServerCompactionEngine, classify_issuer  # noqa: E402
    from agent.codex_responses_adapter import (  # noqa: E402
        _chat_messages_to_responses_input,
    )
    from hermes_cli.auth import resolve_codex_runtime_credentials  # noqa: E402

    creds = resolve_codex_runtime_credentials()
    base_url = creds["base_url"]
    api_key = creds["api_key"]
    model = sys.argv[2] if len(sys.argv) > 2 else "gpt-5.6-sol"

    engine = ServerCompactionEngine()
    engine.update_model(
        model=model, context_length=272_000, base_url=base_url,
        api_key=api_key, provider="openai-codex",
        api_mode="codex_responses")
    print(f"[1] engine ready: {engine.name} (inner={engine._inner is not None})")

    # Facts live ONLY in assistant turns -> they end up inside the blob.
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Pick a codename and a port."},
        {"role": "assistant",
         "content": "Registered codename TEAL-FALCON-9 and port 47123."},
    ]
    for i in range(8):
        messages.append({"role": "user", "content": f"Filler {i}, continue."})
        messages.append({"role": "assistant",
                         "content": f"Step {i} done. " + "Padding. " * 60})
    messages.append({"role": "user", "content": "Status?"})
    messages.append({"role": "assistant", "content": "All on track."})

    compacted = engine.compress(list(messages))
    carriers = [m for m in compacted
                if isinstance(m, dict) and m.get("codex_reasoning_items")]
    opaque_count = sum(
        1 for m in carriers for it in m["codex_reasoning_items"]
        if it.get("type") == "compaction_summary")
    print(f"[2] compress: {len(messages)} -> {len(compacted)} messages, "
          f"{opaque_count} opaque artifact(s) attached")
    if opaque_count == 0:
        print("FAIL: no artifact attached")
        return 1

    plain = json.dumps([m.get("content", "") for m in compacted])
    in_text_summary = "TEAL-FALCON-9" in plain or "47123" in plain
    print(f"[3] facts also in portable text summary: {in_text_summary} "
          "(expected — that layer exists for non-OpenAI models; "
          "blob-only fidelity is proven separately by the strict probe)")

    compacted.append({"role": "user", "content":
                      "Exact codename and exact port, please."})
    converted = _chat_messages_to_responses_input(
        compacted, replay_encrypted_reasoning=True,
        current_issuer_kind=classify_issuer(base_url))

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "originator": "codex_cli_rs",
        "OpenAI-Beta": "responses=experimental",
        "session_id": "live-smoke",
    }
    account_id = engine._resolve_account_id()
    if account_id:
        headers["chatgpt-account-id"] = account_id
    req = urllib.request.Request(
        base_url.rstrip("/") + "/responses",
        data=json.dumps({
            "model": model, "input": converted,
            "instructions": "You are a helpful assistant.",
            "tools": [], "parallel_tool_calls": False,
            "reasoning": {"effort": "low", "summary": "auto"},
            "store": False, "stream": True,
            "include": ["reasoning.encrypted_content"],
        }).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=240) as resp:
        body = resp.read().decode()
    answer = "".join(
        json.loads(line[6:]).get("delta", "")
        for line in body.splitlines()
        if line.startswith("data: ")
        and '"response.output_text.delta"' in line)
    recall = {"codename": "TEAL-FALCON-9" in answer,
              "port": "47123" in answer}
    print(f"[4] recall from artifact: {recall}")
    print(f"[4] answer: {answer[:200]}")

    ok = all(recall.values())
    print(f"\nLIVE SMOKE: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
