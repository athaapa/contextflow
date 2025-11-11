"""
Implements message summarization techniques
"""

from typing import List, Dict
from contextflow.utils.llm import LLMClient


class MessageCompactor:
    def __init__(self, model: str):
        """
        Initialize the MessageCompactor.

        Args:
            model: The LLM provider to use for summarization (e.g., "gemini", "groq").
        """
        self.llm = LLMClient(model)

    def summarize(
        self,
        messages_to_summarize: List[Dict[str, str]],
        max_token_count: int = 500,
    ) -> str:
        """
        Summarizes a list of messages.

        Args:
            messages_to_summarize: The list of messages to compress
            agent_goal: The overall goal to guide the summary
            max_token_count: The target length for the final summary

        Returns:
            summaries: A single string containing the dense summary.
        """
        return self._simple_summarize(messages_to_summarize, max_token_count)

    def _simple_summarize(
        self,
        messages_to_summarize: List[Dict[str, str]],
        max_token_count: int,
    ):
        """
        Summarizes a list of messages using a simple LLM summary.

        Args:
            messages_to_summarize: The list of messages to compress
            agent_goal: The overall goal to guide the summary
            max_token_count: The target length for the final summary

        Returns:
            summaries: A single string containing the dense summary.
        """
        if not messages_to_summarize:
            return ""

        if len(messages_to_summarize) == 1:
            return messages_to_summarize[0]["content"]

        conversation_text = self._format_messages(messages_to_summarize)

        try:
            summary = self.llm.summarize_text(
                source=conversation_text,
                max_tokens=max_token_count,
            )

            return summary.strip()
        except Exception as e:
            # Fallback: return a simple concatenation
            print(f"Warning: Summarization failed ({e}). Using fallback.")
            return self._fallback_summary(messages_to_summarize)

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        Format messages into a readable conversation string.

        Args:
            messages: List of message dictionaries with "role" and "content" keys.

        Returns:
            Formatted string with each message on a new line in "Role: content" format.
        """
        formatted = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            formatted.append(f"{role.capitalize()}: {content}")

        return "\n".join(formatted)

    def _fallback_summary(self, messages: List[Dict[str, str]]) -> str:
        """
        Simple fallback summary if LLM summarization fails.

        Args:
            messages: List of message dictionaries with "role" and "content" keys.

        Returns:
            Concatenated message contents joined with " ... ".
        """
        # Just concatenate the messages with "..." between them
        contents = [msg.get("content", "") for msg in messages]
        return " ... ".join(contents)

    def _hierarchical_summarize(
        self,
        messages_to_summarize: List[Dict[str, str]],
        agent_goal: str,
        max_token_count: int,
    ):
        """
        Summarizes a list of messages using the map-reduce method.

        Args:
            messages_to_summarize: The list of messages to compress
            agent_goal: The overall goal to guide the summary
            max_summary_tokens: The target length for the final summary

        Returns:
            summaries: A single string containing the dense summary.
        """
        raise NotImplementedError(
            "Hierarchical summarization not yet implemented. "
            "Use simple_summarize for now."
        )
