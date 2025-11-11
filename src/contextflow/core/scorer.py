"""
Message relevance and utility scoring
"""

from typing import List, Dict
from contextflow.utils.llm import LLMClient
import asyncio


class MessageScorer:
    def __init__(self, model: str):
        """
        Initialize the MessageScorer.

        Args:
            model: The LLM provider to use for scoring (e.g., "gemini", "groq").
        """
        self.llm = LLMClient(model)

    def _create_batches(
        self, messages: List[Dict[str, str]], batch_size: int = 20
    ):
        """
        Split messages into batches for parallel processing.

        Args:
            messages: List of message dictionaries to batch.
            batch_size: Maximum number of messages per batch. Defaults to 20.

        Returns:
            List of message batches, each containing up to batch_size messages.
        """
        if len(messages) < batch_size:
            return [messages]

        return [
            messages[i : i + batch_size]
            for i in range(0, len(messages), batch_size)
        ]

    def score_messages(
        self, messages: List[Dict[str, str]], goal: str
    ) -> List[float]:
        """
        Score messages synchronously based on relevance to the agent's goal.

        Args:
            messages: List of message dictionaries with "role" and "content" keys.
            goal: The goal of the agent to guide relevance scoring.

        Returns:
            List of relevance scores (0-10) corresponding to each message.
        """
        return asyncio.run(self.score_all(messages, goal))

    async def score_all(
        self, messages: List[Dict[str, str]], goal: str
    ) -> List[float]:
        """Scores messages based on how relevant they are to the agent's goal.

        Args:
            messages: A list of messages
            goal: The goal of the agent
        Returns:
            A list scores such that scores[i] is the relevancy score of messages[i]
        """

        scores = []

        batches = self._create_batches(messages, 20)

        tasks = [
            self.llm.score_batch_async(goal=goal, batch=batch, max_tokens=400)
            for batch in batches
        ]

        results = await asyncio.gather(*tasks)

        for x in results:
            scores.extend(x)

        for i in range(len(scores) - 1, max(-1, len(scores) - 6), -1):
            recency_bonus = 1.0
            scores[i] += recency_bonus

        return scores
