"""
Token counting utilities
"""

from typing import List, Dict
import tiktoken

# Cache encoders to avoid reloading
_ENCODERS = {}


def get_encoder(model: str = "gpt-4") -> tiktoken.Encoding:
    """Get tiktoken encoder for a model

    Args:
        model: Model name (gpt-4, gpt-3.5-turbo, claude-3, etc.)

    Returns:
        A tiktoken encoder
    """
    if model not in _ENCODERS:
        try:
            # Most models use cl100k_base encoding
            if model.startswith("gpt-4") or model.startswith("gpt-3.5"):
                _ENCODERS[model] = tiktoken.encoding_for_model(model)
            else:
                # Default to cl100k_base for other models (Claude, etc.)
                _ENCODERS[model] = tiktoken.get_encoding("cl100k_base")
        except KeyError:
            # Fallback to cl100k_base
            _ENCODERS[model] = tiktoken.get_encoding("cl100k_base")

    return _ENCODERS[model]


def count_tokens(messages: List[Dict[str, str]], model: str = "gpt-4") -> int:
    """
    Count tokens in a list of messages

    Args:
        messages: List of message dicts [{"role": "user", "content": "..."}]
        model: Model to count tokens for

    Returns:
        Total token count
    """
    encoder = get_encoder(model)

    total_tokens = 0

    for message in messages:
        # Count tokens in role (e.g., "user", "assistant", "system")
        total_tokens += len(encoder.encode(message.get("role", "")))

        # Count tokens in content
        total_tokens += len(encoder.encode(message.get("content", "")))

        # Add overhead per message (OpenAI adds ~4 tokens per message for formatting)
        total_tokens += 4

    # Add overhead for priming the response
    total_tokens += 3

    return total_tokens


def count_tokens_simple(text: str, model: str = "gpt-4") -> int:
    """
    Count tokens in a simple text string

    Args:
        text: Input text
        model: Model to count tokens for

    Returns:
        Token count
    """
    encoder = get_encoder(model)
    return len(encoder.encode(text))
