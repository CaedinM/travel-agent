"""Small, explicit token-price map used by flight pipeline evaluations.

Prices are standard, uncached API rates in USD per million tokens. They intentionally
live in code rather than an evaluation spreadsheet so every JSON report records the
same arithmetic that produced its estimated model cost.
"""

# Standard, uncached API prices in USD per one million tokens.
INPUT_TOKEN_PRICES: dict[str, float] = {
    "jev-1.13.0": 0.042,
    "openai:gpt-5-mini": 0.25,
}

OUTPUT_TOKEN_PRICES: dict[str, float] = {
    "jev-1.13.0": 0.0,
    "openai:gpt-5-mini": 2.0,
}


def estimate_token_cost_usd(
    model: str, input_tokens: int | None, output_tokens: int | None
) -> float | None:
    """Estimate standard token cost, returning None when usage or price is unknown."""
    input_price = INPUT_TOKEN_PRICES.get(model)
    output_price = OUTPUT_TOKEN_PRICES.get(model)
    if input_price is None or output_price is None or input_tokens is None or output_tokens is None:
        return None
    return round(
        (input_tokens * input_price + output_tokens * output_price) / 1_000_000,
        8,
    )
