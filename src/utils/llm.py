"""
Local LLM client using Ollama
"""

from google import genai
from google.genai import types
import json
import os


class LLMClient:
    """Lightweight LLM utility"""

    def __init__(self):
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model_name = "gemini-2.5-flash-lite"

    def generate_text(
        self, prompt: str, max_tokens: int = 500, temperature: float = 0.3
    ) -> str:
        """
        Generate text from a prompt (for summarization)

        Args:
            prompt: The prompt to send
            max_tokens: Maximum tokens in response
            temperature: Randomness (0=deterministic, 1=creative)

        Returns:
            Generated text
        """

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                ),
            )

            return response.text

        except Exception as e:
            raise Exception(f"LLM generation failed: {e}")

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
