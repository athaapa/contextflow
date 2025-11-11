"""
Message relevance and utility scoring
"""

from typing import List, Dict
from google.genai import types
from contextflow.utils.llm import LLMClient
import json
import asyncio


class MessageScorer:
    def __init__(self, model: str):
        """
        Initialize the MessageScorer.

        Args:
            model: The LLM provider to use for scoring (e.g., "gemini", "groq").
        """
        self._embedding_cache = {}
        self._llm_client = LLMClient(model)

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
        self, messages: List[Dict[str, str]], agent_goal: str
    ) -> List[float]:
        """
        Score messages synchronously based on relevance to the agent's goal.

        Args:
            messages: List of message dictionaries with "role" and "content" keys.
            agent_goal: The goal of the agent to guide relevance scoring.

        Returns:
            List of relevance scores (0-10) corresponding to each message.
        """
        return asyncio.run(self.score_all(messages, agent_goal))

    async def score_all(
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

        batches = self._create_batches(messages, 20)

        tasks = [
            self._score_batch_async(batch, agent_goal) for batch in batches
        ]

        results = await asyncio.gather(*tasks)

        for x in results:
            scores.extend(x)

        for i in range(len(scores) - 1, max(-1, len(scores) - 6), -1):
            recency_bonus = 1.0
            scores[i] += recency_bonus

        return scores

    def score_utility(self, text: str, goal: str) -> float:
        """
        Score message utility using an LLM

        Args:
            text: Message content
            goal: Agent goal for context

        Returns:
            Utility score 0-10
        """

        prompt = f"""Rate message relevance to goal (0-10 scale):

        Goal: {goal}

        Scoring guide:
        • 8-10: Specific facts, numbers, IDs, errors ("Error: timeout line 42" = 10)
        • 4-7: Questions, partial info ("Can you check status?" = 6)
        • 0-3: Acknowledgments, greetings ("Thanks!" = 1)

        Message: "{text}"

        Score (number only):"""

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
                response_schema={
                    "type": "object",
                    "properties": {
                        "score": {
                            "type": "number",
                            "description": "Rating from 0-10",
                            "minimum": 0,
                            "maximum": 10,
                        }
                    },
                    "required": ["score"],
                },
            ),
        )
        result = json.loads(response.text)
        score = result["score"]

        return score

    async def _score_batch_async(
        self, batch: List[Dict[str, str]], goal: str
    ) -> List[float]:
        """
        Score a batch of messages asynchronously using an LLM.

        Args:
            batch: List of message dictionaries with "role" and "content" keys.
            goal: Agent goal for context to guide relevance scoring.

        Returns:
            List of utility scores (0-10) for each message in the batch.
        """

        formatted_messages = ""
        for i, msg in enumerate(batch, 1):
            role = msg.get("role", "unknown").capitalize()
            content = msg.get("content", "")
            # Truncate long messages to save prompt tokens?
            # formatted_messages += f"{i}. [{role}] {content[:200]}\n"
            formatted_messages += f"{i}. [{role}] {content}\n"

        prompt = f"""Rate message relevance to goal (0-10 scale):

        Goal: {goal}

        Scoring guide (err on the LOW side):
        9-10: ONLY critical facts/errors with specific details ("NullPointerException line 42" = 10)
        6-8: Important context, questions, partial info ("Can you check?" = 7)
        3-5: Minor details, acknowledgments ("I see" = 4)
        0-2: Pure filler, greetings, "ok", "thanks" (= 1)


        MESSAGES TO RATE:
        {formatted_messages}

        Return a JSON array with one score per message in order
        """
        response_schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "message_index": {
                        "type": "integer",
                        "description": "The message number (1-indexed)",
                    },
                    "score": {
                        "type": "number",
                        "description": "Utility score from 0-10",
                        "minimum": 0,
                        "maximum": 10,
                    },
                },
                "required": ["message_index", "score"],
            },
        }

        response = await self._llm_client.generate_content_async(
            prompt,
            types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
                response_schema=response_schema,
            ),
        )

        scores_data = json.loads(response.text)

        # scores = scores_data["scores"]
        scores = self._extract_scores_from_json(scores_data, len(batch))

        return scores

    def _extract_scores_from_json(
        self, scores_data: List[Dict], expected_count: int
    ) -> List[float]:
        """
        Extracts scores from the JSON array response.

        Args:
            scores_data: List of {"message_index": 1, "score": 8.5} objects
            expected_count: Number of messages we scored

        Returns:
            List of scores in the correct order
        """

        # Create a dict for quick lookup
        score_dict = {}
        for item in scores_data:
            idx = item.get("message_index")
            score = item.get("score", 5.0)
            if idx is not None:
                score_dict[idx] = max(0.0, min(10.0, float(score)))

        # Build the final list in order (1-indexed to 0-indexed)
        final_scores = []
        for i in range(1, expected_count + 1):
            final_scores.append(
                score_dict.get(i, 5.0)
            )  # Default to 5.0 if missing

        return final_scores
