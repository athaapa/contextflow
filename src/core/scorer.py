"""
Message relevance and utility scoring
"""

from typing import List, Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from src.utils.llm import LLMClient
import numpy as np


class MessageScorer:
    def __init__(self):
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self._embedding_cache = {}
        self._llm_client = LLMClient()

    def _encode_text(self, text: str) -> np.ndarray:
        """Convert text to a vector embedding

        Args:
            text (str): Text to encode
        Returns:
            A 384-dimensional numpy array representing the text encoding
        """

        if text in self._embedding_cache:
            return self._embedding_cache[text]

        embedding = self.embedding_model.encode(
            text,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        self._embedding_cache[text] = embedding

        return embedding

    def _calculate_relevance(
        self, message_text: str, goal_embedding: np.ndarray
    ):
        """Calculate semantic relevance (0.0 to 1.0)

        Args:
            message_text: The message content
            goal_embedding: Pre-computed goal vector

        Returns:
            Similarity score (0.0 = unrelated, 1.0 = identical)
        """

        message_embedding = self._encode_text(message_text)

        # Reshape for sklearn (needs 2D arrays)
        goal_2d = goal_embedding.reshape(1, -1)
        message_2d = message_embedding.reshape(1, -1)

        # Calculate cosine similarity
        similarity = cosine_similarity(goal_2d, message_2d)[0][0]

        # Clamp to [0, 1] (cosine similarity can be negative)
        similarity = max(0.0, similarity)

        return similarity

    def score_messages(
        self, messages: List[Dict[str, str]], agent_goal: str
    ) -> List[float]:
        """Scores messages based on how relevant they are to the agent's goal.

        Args:
            messages: A list of messages
            agent_goal: The goal of the agent
        Returns:
            A list scores such that scores[i] is the relevancy score of messages[i]
        """

        scores = []
        # goal_embedding = self._encode_text(agent_goal)

        for i, message in enumerate(messages):
            content = message.get("content", "")

            if not content.strip():
                scores.append(0.0)
                continue

            # relevance_score = self._calculate_relevance(content, goal_embedding)
            utility_score = self._calculate_utility_heuristic(
                content, agent_goal
            )

            recency_bonus = 1.0 if i >= len(messages) - 5 else 0.0

            composite = utility_score + recency_bonus
            scores.append(composite)

        return scores

    def _calculate_utility_heuristic(self, text: str, goal: str) -> float:
        """Returns a utility score (0-10 scale)

        Args:
            text: String to evaluate the utility of
        Returns:
            Utility score of `text` between 0 and 10
        """
        text = text.strip()
        text_lower = text.lower()

        return self._llm_client.score_utility(text_lower, goal)
