"""
Token counting utilities
"""

from typing import List, Dict


def count_tokens(messages: List[Dict[str, str]]) -> int:
    """
    Count tokens in a list of messages

    Args:
        messages: List of message dicts [{"role": "user", "content": "..."}]

    Returns:
        Total token count
    """

    # Hueristic. Not exact

    total = 0

    for message in messages:
        total += len(message["content"])

    return total // 4
