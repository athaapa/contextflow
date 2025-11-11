from contextflow.utils.providers.base import LLMProvider
from google.genai import Client, types
from typing import List, Dict
import json


MODEL_NAME = "gemini-2.5-flash-lite"


class LLM(LLMProvider):
    def __init__(self, client: Client):
        self.client = client

    def summarize_text(self, source: str, max_tokens: int):
        prompt = f"""You are summarizing a conversation to preserve key information while reducing length.

        Conversation:
        {source}

        Instructions:
        - Create a dense, information-rich summary
        - Preserve all critical facts, names, numbers, and decisions
        - Remove pleasantries and redundant information
        - Target length: approximately {max_tokens} tokens
        - Write in third person (e.g., "User reported X. Agent confirmed Y.")
        - Be extremely concise. Proper English is not necessary. Convey the utmost with the least.

        Summary:"""

        response = self.client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
            ),
        )

        return response.text

    def score_message(self, goal: str, message: str):
        prompt = f"""Rate message relevance to goal (0-10 scale):

        Goal: {goal}

        Scoring guide:
        • 8-10: Specific facts, numbers, IDs, errors ("Error: timeout line 42" = 10)
        • 4-7: Questions, partial info ("Can you check status?" = 6)
        • 0-3: Acknowledgments, greetings ("Thanks!" = 1)

        Message: "{message}"

        Score (number only):"""
        return self.client.models.generate_content(
            model=MODEL_NAME,
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

    async def score_batch_async(
        self,
        goal: str,
        batch: List[Dict[str, str]],
        max_tokens: int,
    ):
        formatted_messages = ""
        for i, msg in enumerate(batch, 1):
            role = msg.get("role", "unknown").capitalize()
            content = msg.get("content", "")
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

        response = await self.client.aio.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
                response_schema=response_schema,
                max_output_tokens=max_tokens,
            ),
        )

        scores_data = json.loads(response.text)
        scores = self._extract_scores_from_json(scores_data, len(batch))

        return scores
