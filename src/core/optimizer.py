from src.core.compactor import MessageCompactor
from src.core.scorer import MessageScorer
from typing import List, Dict, Union, Literal
from src.core.strategies import balanced_strategy


class ContextOptimizer:
    def __init__(self):
        self.message_compactor = MessageCompactor()
        self.message_scorer = MessageScorer()

    def optimize(
        self,
        messages: List[Dict[str, str]],
        agent_goal: str,
        strategy: Union[
            Literal["conservative"], Literal["balanced"], Literal["aggressive"]
        ] = Literal["balanced"],
        max_token_count: int = 500,
    ):
        scores = self.message_scorer.score_messages(
            messages=messages, agent_goal=agent_goal
        )
        print(scores)

        optimized = None
        if strategy == "balanced":
            optimized = balanced_strategy(
                messages, scores, max_token_count, self.message_compactor
            )

        return optimized
