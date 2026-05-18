"""Shared tiktoken encoder for consistent token counting across all injection surfaces."""

import tiktoken

_enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count tokens using the cl100k_base encoding."""
    return len(_enc.encode(text))
