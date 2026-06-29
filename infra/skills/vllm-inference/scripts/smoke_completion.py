#!/usr/bin/env python3
# Purpose: send one chat/completions request to a running vLLM OpenAI-compatible
#          server and print the reply — end-to-end serving smoke test.
# Usage: ./smoke_completion.py --model <name> [--base-url URL] [--prompt TEXT]
import argparse
import sys

# vLLM ignores the key but the openai client requires a non-empty string.
DUMMY_KEY = "EMPTY"
# vLLM's default OpenAI server bind; --port maps to this path.
DEFAULT_BASE_URL = "http://localhost:8000/v1"
# Keep the smoke response short — we verify serving works, not generation quality.
MAX_TOKENS = 32


def main() -> int:
    parser = argparse.ArgumentParser(description="vLLM serving smoke test")
    parser.add_argument("--model", required=True, help="served model name (--served-model-name)")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--prompt", default="Say 'ok' in one word.")
    args = parser.parse_args()

    try:
        from openai import OpenAI
    except ImportError:
        print("error: 'openai' package not installed (pip install openai)", file=sys.stderr)
        return 2

    client = OpenAI(base_url=args.base_url, api_key=DUMMY_KEY)
    try:
        resp = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": args.prompt}],
            max_tokens=MAX_TOKENS,
        )
    except Exception as exc:  # connection refused, 404 model, etc.
        print(f"error: request to {args.base_url} failed: {exc}", file=sys.stderr)
        return 1

    if not resp.choices:
        print("error: server returned no choices", file=sys.stderr)
        return 1

    print(resp.choices[0].message.content)
    return 0


if __name__ == "__main__":
    sys.exit(main())
