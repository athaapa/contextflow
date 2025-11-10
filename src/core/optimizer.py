from src.core.compactor import MessageCompactor
from src.core.scorer import MessageScorer
from typing import List, Dict, Union, Literal
from src.core.strategies import balanced_strategy
from src.utils.tokenizer import count_tokens


class ContextFlow:
    def __init__(self):
        self.message_compactor = MessageCompactor()
        self.message_scorer = MessageScorer()

    def optimize(
        self,
        messages: List[Dict[str, str]],
        agent_goal: str,
        max_token_count: int = 500,
    ):
        scores = self.message_scorer.score_messages(
            messages=messages, agent_goal=agent_goal
        )

        optimized = None
        optimized = balanced_strategy(
            messages, scores, max_token_count, self.message_compactor
        )

        tokens_before = count_tokens(messages)
        tokens_after = count_tokens(optimized)
        reduction_pct = (1 - (tokens_after / tokens_before)) * 100

        return {
            "messages": optimized,
            "analytics": {
                "tokens_after": tokens_after,
                "reduction_pct": reduction_pct,
                "tokens_saved": tokens_before - tokens_after,
            },
        }
