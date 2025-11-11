"""
Local LLM client using Gemini
"""

from google import genai
from google.genai import types
from groq import Groq
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor


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
            self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            self.model_name = "gemini-2.5-flash-lite"
        elif provider == "groq":
            self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            self.model_name = "llama-3.3-70b-versatile"
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def summarize_text(
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

    def generate_content(
        self, prompt: str, config: types.GenerateContentConfig
    ):
        """
        Generate content synchronously using the configured LLM.

        Args:
            prompt: The text prompt to send to the LLM.
            config: GenerateContentConfig object specifying generation parameters.

        Returns:
            Response object from the LLM containing the generated content.
        """
        return self.client.models.generate_content(
            model=self.model_name, contents=prompt, config=config
        )

    async def generate_content_async(
        self, prompt: str, config: types.GenerateContentConfig
    ):
        """
        Generate content asynchronously using the configured LLM.

        Args:
            prompt: The text prompt to send to the LLM.
            config: GenerateContentConfig object specifying generation parameters.

        Returns:
            Response object from the LLM containing the generated content.
        """
        if self.provider == "gemini":
            return await self.client.aio.models.generate_content(
                model=self.model_name, contents=prompt, config=config
            )
        else:
            print("groq")
            loop = asyncio.get_event_loop()
            executor = ThreadPoolExecutor(max_workers=1)

            def _sync_call():
                response = self.groq_client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=500,
                    response_format={"type": "json_object"},
                )
                return response.choices[0].message.content

            text = await loop.run_in_executor(executor, _sync_call)

            class GroqResponse:
                def __init__(self, text):
                    self.text = text

            return GroqResponse(text)
