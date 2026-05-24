"""OpenRouter token prices per model, USD per 1M tokens.

Used by ``metrics.cost_usd`` to convert token counts into dollars.

Keep this table tight — only models we actually run. Update when OpenRouter
changes prices (rare) or when we onboard a new model. If a model is missing,
``cost_usd`` returns None rather than guessing, so a missing entry surfaces
as an explicit unknown in the report rather than a silent zero.
"""

from __future__ import annotations

# (input $/1M, output $/1M)
_PRICES: dict[str, tuple[float, float]] = {
    "deepseek/deepseek-v4-flash": (0.05, 0.25),
    "deepseek/deepseek-v4-flash:free": (0.0, 0.0),
    "deepseek/deepseek-v4-pro": (0.27, 1.10),
    "anthropic/claude-opus-4-7": (15.0, 75.0),
    "anthropic/claude-sonnet-4-6": (3.0, 15.0),
    "google/gemini-3-pro-preview": (1.25, 5.0),
}


def price_per_million(model: str) -> tuple[float, float] | None:
    return _PRICES.get(model)


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    p = _PRICES.get(model)
    if p is None:
        return None
    in_price, out_price = p
    return (input_tokens / 1_000_000) * in_price + (output_tokens / 1_000_000) * out_price
