from contextflow.core.compactor import MessageCompactor
from contextflow.core.scorer import MessageScorer
from typing import List, Dict
from contextflow.core.strategies import balanced_strategy
from contextflow.utils.tokenizer import count_tokens
import time


class ContextFlow:
    def __init__(
        self,
        scoring_model: str = "gemini",
        summarizing_model: str = "gemini",
    ):
        """
        Initialize the ContextFlow optimizer.

        Args:
            scoring_model: The LLM provider to use for scoring message relevance.
                          Options: "gemini", "groq". Defaults to "gemini".
            summarizing_model: The LLM provider to use for summarizing messages.
                              Options: "gemini", "groq". Defaults to "gemini".
        """
        self.message_compactor = MessageCompactor(model=summarizing_model)
        self.message_scorer = MessageScorer(model=scoring_model)

    def optimize(
        self,
        messages: List[Dict[str, str]],
        goal: str,
        max_token_count: int = 500,
    ):
        """
        Optimize a conversation by reducing token count while preserving important information.

        Args:
            messages: List of message dictionaries with "role" and "content" keys.
            goal: The goal or purpose of the agent to guide relevance scoring.
            max_token_count: Maximum number of tokens allowed in the optimized output.
                           Defaults to 500.

        Returns:
            Dictionary containing:
                - "messages": Optimized list of messages
                - "analytics": Dictionary with optimization metrics:
                    - "tokens_after": Token count after optimization
                    - "reduction_pct": Percentage reduction in tokens
                    - "tokens_saved": Number of tokens saved
                    - "time_taken_ms": Time taken for optimization in milliseconds
        """
        start_time = time.time_ns() // 1_000_000

        scores = self.message_scorer.score_messages(
            messages=messages, goal=goal
        )

        optimized = balanced_strategy(
            messages, scores, max_token_count, self.message_compactor
        )

        tokens_before = count_tokens(messages)
        tokens_after = count_tokens(optimized)
        reduction_pct = ((tokens_before - tokens_after) / tokens_before) * 100

        now = time.time_ns() // 1_000_000

        return {
            "messages": optimized,
            "analytics": {
                "tokens_after": tokens_after,
                "reduction_pct": reduction_pct,
                "tokens_saved": tokens_before - tokens_after,
                "time_taken_ms": now - start_time,
            },
        }
