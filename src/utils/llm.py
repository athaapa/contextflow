"""
Local LLM client using Ollama
"""

from google import genai
from google.genai import types
import json
import os


api_key = os.getenv("GEMINI_API_KEY")


class LLMClient:
    """Lightweight LLM for utility scoring"""

    def __init__(self):
        self.client = genai.Client(api_key=api_key)

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
            model="gemini-2.5-flash",
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
