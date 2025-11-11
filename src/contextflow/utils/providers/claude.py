from anthropic import Anthropic
import asyncio
from contextflow.utils.providers.base import LLMProvider
from typing import List, Dict
import json
import re


MODEL_NAME = "claude-haiku-4-5-20251001"


class LLM(LLMProvider):
    def __init__(self, client: Anthropic):
        self.client = client

    def summarize_text(self, source: str, max_tokens: int) -> str:
        prompt = f"""You are summarizing a conversation to preserve key information while reducing length.

        Conversation:
        {source}

        Instructions:
        - Create a dense, information-rich summary
        - Preserve all critical facts, names, numbers, and decisions
        - Remove pleasantries and redundant information
        - Target length: approximately {max_tokens} tokens
        - Write in third person (e.g., "User reported X. Agent confirmed Y.")
        - Be extremely concise. Proper English is not necessary. Convey the utmost and the least amount of characters.

        Summary:"""

        # print(max_tokens)

        response = self.client.messages.create(
            model=MODEL_NAME,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        return response.content[0].text

    def score_message(self, goal: str, message: str) -> Dict:
        prompt = f"""Rate message relevance to goal (0-10 scale):

        Goal: {goal}

        Scoring guide:
        • 8-10: Specific facts, numbers, IDs, errors ("Error: timeout line 42" = 10)
        • 4-7: Questions, partial info ("Can you check status?" = 6)
        • 0-3: Acknowledgments, greetings ("Thanks!" = 1)

        Message: "{message}"

        Output in JSON format with key: "score" (integer).

        Score (number only):"""
        response = self.client.messages.create(
            model=MODEL_NAME,
            max_tokens=10,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )
        return response.content.text

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
        self, goal: str, batch: List[Dict[str, str]], max_tokens: int
    ):
        characters = 0
        formatted_messages = ""
        for i, msg in enumerate(batch, 1):
            role = msg.get("role", "unknown").capitalize()
            content = msg.get("content", "")
            formatted_messages += f"{i}. [{role}] {content}\n"
            characters += len(content)

        prompt = f"""Rate message relevance to goal (0-10 scale):

        Goal: {goal}

        Scoring guide (err on the LOW side):
        9-10: ONLY critical facts/errors with specific details ("NullPointerException line 42" = 10)
        6-8: Important context, questions, partial info ("Can you check?" = 7)
        3-5: Minor details, acknowledgments ("I see" = 4)
        0-2: Pure filler, greetings, "ok", "thanks" (= 1)

        Most messages should score between 3-6. Be HARSH but FAIR.

        MESSAGES TO RATE:
        {formatted_messages}

        DO NOT RETURN ANYTHING OTHER THAN A JSON ARRAY.
        
        Return ONLY a JSON array with one score per message in order:
        """

        def blocking_call():
            return self.client.messages.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=0,
                max_tokens=max_tokens,
            )

        response = await asyncio.to_thread(blocking_call)

        raw_text = response.content[0].text.strip()

        cleaned = re.sub(
            r"^```json\\?\n?|```$", "", raw_text, flags=re.IGNORECASE
        )
        cleaned = cleaned.replace("\\n", "\n")  # unescape newlines

        # print(raw_text, cleaned)
        scores = json.loads(cleaned)

        return scores
