"""
Implements message summarization techniques
"""

from typing import List, Dict
from src.utils.llm import LLMClient


class MessageCompactor:
    def __init__(self):
        self.llm = LLMClient()

    def summarize(
        self,
        messages_to_summarize: List[Dict[str, str]],
        agent_goal: str,
        max_summary_tokens: int = 500,
    ) -> str:
        """
        Summarizes a list of messages.

        Args:
            messages_to_summarize: The list of messages to compress
            agent_goal: The overall goal to guide the summary
            max_summary_tokens: The target length for the final summary

        Returns:
            summaries: A single string containing the dense summary.
        """
        pass

    def _simple_summarize(
        self,
        messages_to_summarize: List[Dict[str, str]],
        agent_goal: str,
        max_summary_tokens: int,
    ):
        """
        Summarizes a list of messages using a simple LLM summary.

        Args:
            messages_to_summarize: The list of messages to compress
            agent_goal: The overall goal to guide the summary
            max_summary_tokens: The target length for the final summary

        Returns:
            summaries: A single string containing the dense summary.
        """
        pass

    def _hierarchical_summarize(
        self,
        messages_to_summarize: List[Dict[str, str]],
        agent_goal: str,
        max_summary_tokens: int,
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
        pass
