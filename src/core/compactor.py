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

        prompt = f"""You are summarizing a conversation to preserve key information while reducing length.

        Conversation:
        {conversation_text}

        Instructions:
        - Create a dense, information-rich summary
        - Preserve all critical facts, names, numbers, and decisions
        - Remove pleasantries and redundant information
        - Target length: approximately {max_token_count} tokens
        - Write in third person (e.g., "User reported X. Agent confirmed Y.")

        Summary:"""

        try:
            summary = self.llm.generate_text(
                prompt,
                max_tokens=max_token_count,
                temperature=0.3,  # Low temperature for factual summary
            )

            return summary.strip()
        except Exception as e:
            # Fallback: return a simple concatenation
            print(f"Warning: Summarization failed ({e}). Using fallback.")
            return self._fallback_summary(messages_to_summarize)

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format messages into a readable conversation"""

        formatted = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            formatted.append(f"{role.capitalize()}: {content}")

        return "\n".join(formatted)

    def _fallback_summary(self, messages: List[Dict[str, str]]) -> str:
        """Simple fallback if LLM summarization fails"""

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
