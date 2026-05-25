#!/usr/bin/env python3
# Purpose: Smoke-test a running vLLM OpenAI-compatible server with a minimal
#          chat-completion request; assert non-empty response and print throughput.
# Usage:   python validate_openai_endpoint.py [--url URL] [--model MODEL] [--timeout SECS]

import argparse
import json
import time
import urllib.error
import urllib.request
import sys

DEFAULT_URL     = "http://localhost:8000"   # vLLM default bind address
DEFAULT_TIMEOUT = 30                        # seconds before giving up on hung server
TEST_PROMPT     = "Reply with one word: hello"  # minimal prompt — keeps token count low


def parse_args():
    p = argparse.ArgumentParser(description="vLLM endpoint smoke test")
    p.add_argument("--url",     default=DEFAULT_URL,     help="Base URL of vLLM server")
    p.add_argument("--model",   default=None,            help="Model name (auto-detected if omitted)")
    p.add_argument("--timeout", default=DEFAULT_TIMEOUT, type=int, help="Request timeout in seconds")
    return p.parse_args()


def get_first_model(base_url, timeout):
    req = urllib.request.Request(f"{base_url}/v1/models")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    models = data.get("data", [])
    if not models:
        print("error: /v1/models returned no models", file=sys.stderr)
        sys.exit(1)
    return models[0]["id"]


def chat_complete(base_url, model, timeout):
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": TEST_PROMPT}],
        "max_tokens": 16,   # tiny ceiling — smoke test only needs a non-empty reply
    }).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read())
    elapsed = time.monotonic() - t0
    return body, elapsed


if __name__ == "__main__":
    args = parse_args()

    try:
        model = args.model or get_first_model(args.url, args.timeout)
    except urllib.error.URLError as e:
        print(f"error: cannot reach {args.url} — {e.reason}", file=sys.stderr)
        sys.exit(1)

    print(f"model : {model}")

    try:
        body, elapsed = chat_complete(args.url, model, args.timeout)
    except urllib.error.URLError as e:
        print(f"error: chat completion failed — {e.reason}", file=sys.stderr)
        sys.exit(1)

    content = body.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    if not content:
        print("error: response content is empty", file=sys.stderr)
        sys.exit(1)

    usage      = body.get("usage", {})
    out_tokens = usage.get("completion_tokens", 0)
    throughput = out_tokens / elapsed if elapsed > 0 else float("inf")

    print(f"reply : {content!r}")
    print(f"tokens: {out_tokens} completion tokens in {elapsed:.2f}s ({throughput:.1f} tok/s)")
    print("status: ok")
