"""
Local LLM client using Gemini
"""

from google import genai
from groq import Groq
import os
from anthropic import Anthropic
from openai import OpenAI
from typing import List, Dict

from contextflow.utils.providers import gemini, groq, claude


class LLMClient:
    """Lightweight LLM utility"""

    def __init__(self, provider: str = "gemini"):
        """
        Initialize the LLM client with the specified provider.

        Args:
            provider: The LLM provider to use. Options: "gemini" or "groq".
                     Defaults to "gemini".

        Raises:
            ValueError: If an unknown provider is specified.
        """
        self.provider = provider

        if provider == "gemini":
            self.google_client = genai.Client(
                api_key=os.getenv("GEMINI_API_KEY")
            )
        elif provider == "groq":
            self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        elif provider == "anthropic":
            self.anthropic_client = Anthropic(
                api_key=os.getenv("ANTHROPIC_KEY")
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def summarize_text(self, source: str, max_tokens: int) -> str:
        """
        Generate text from a prompt (for summarization)

        Args:
            prompt: The prompt to send
            max_tokens: Maximum tokens in response
            temperature: Randomness (0=deterministic, 1=creative)

        Returns:
            Generated text
        """
        match self.provider:
            case "gemini":
                return gemini.LLM(self.google_client).summarize_text(
                    source=source,
                    max_tokens=max_tokens,
                )
            case "anthropic":
                return claude.LLM(self.anthropic_client).summarize_text(
                    source=source,
                    max_tokens=max_tokens,
                )

    async def score_batch_async(
        self, goal: str, batch: List[Dict[str, str]], max_tokens: int
    ):
        match self.provider:
            case "gemini":
                return await gemini.LLM(self.google_client).score_batch_async(
                    goal=goal,
                    batch=batch,
                    max_tokens=max_tokens,
                )
            case "anthropic":
                return await claude.LLM(
                    self.anthropic_client
                ).score_batch_async(
                    goal=goal,
                    batch=batch,
                    max_tokens=max_tokens,
                )

        # Fallback
        return [5.0] * len(batch)
