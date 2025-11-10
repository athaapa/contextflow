from typing import List, Union, Literal
from src.utils.tokenizer import count_tokens  # You'll need this
from src.core.compactor import MessageCompactor


def get_strategy(
    strategy_name: Union[
        Literal["conservative"], Literal["balanced"], Literal["aggressive"]
    ],
):
    """Factory to get strategy function"""
    strategies = {
        "conservative": conservative_strategy,
        "balanced": balanced_strategy,
        "aggressive": aggressive_strategy,
    }
    return strategies.get(strategy_name, balanced_strategy)


def conservative_strategy():
    pass


def balanced_strategy(
    messages: List[str],
    scores: List[float],
    max_token_count: int,
    compactor: MessageCompactor,
):
    """Optimizes a conversation (i.e. a list of messages) using a balanced strategy (keep high-scoring, summarize mid, drop low)

    Args:
        messages: List of messages
        scores: List of scores for each message
        max_token_count: Maximum number of tokens allowed
        compactor: Tool for summarizing messages
    Returns:
        Optimized list of messages that is less than max_token_count
    """

    preserve_recent = 5
    recent_messages = messages[-preserve_recent:]
    older_messages = messages[:-preserve_recent]
    older_scores = scores[:-preserve_recent]

    sorted_pairs = sorted(
        zip(older_messages, older_scores),
        key=lambda x: x[1],  # Sort by the score (second element of tuple)
        reverse=True,  # Highest scores first
    )

    optimized = recent_messages.copy()
    current_tokens = count_tokens(optimized)

    # FIX: Check if we're already over budget with just recent messages
    if current_tokens >= max_token_count:
        # Emergency: Even recent messages exceed budget
        # Keep reducing until we fit
        while current_tokens > max_token_count and len(optimized) > 1:
            optimized.pop(0)  # Remove oldest of the recent messages
            current_tokens = count_tokens(optimized)
        return optimized

    # Categorize older messages into buckets
    keep_bucket = []
    summarize_bucket = []

    print("Summarizing...")

    for message, score in sorted_pairs:
        if score >= 7.0:
            keep_bucket.append(message)
        elif score > 4.0:
            summarize_bucket.append(message)
        # score <= 4.0: drop entirely

    # Try to add high-scoring messages one by one
    for message in keep_bucket:
        message_tokens = count_tokens([message])
        if current_tokens + message_tokens <= max_token_count:
            optimized.insert(0, message)  # Add before recent messages
            current_tokens += message_tokens
        else:
            # print(f"adding {message} to summary...")
            # Can't fit this message, add to summarize bucket instead
            summarize_bucket.append(message)

    # print(summarize_bucket)
    if summarize_bucket:
        summary = compactor.summarize(
            summarize_bucket, max_token_count - current_tokens
        )
        summary_message = {
            "role": "system",
            "content": f"Summary of earlier context: {summary}",
        }

        # print(summary_message)

        summary_tokens = count_tokens([summary_message])
        if current_tokens + summary_tokens <= max_token_count:
            optimized.insert(0, summary_message)
            current_tokens += summary_tokens
        # If summary doesn't fit, skip it (rare but possible)

    return optimized


def aggressive_strategy():
    pass
