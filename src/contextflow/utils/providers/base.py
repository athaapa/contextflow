from typing import List, Dict

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    @abstractmethod
    def summarize_text(
        self,
        goal: str,
        batch: List[Dict[str, str]],
        max_tokens: int,
    ) -> str:
        pass

    @abstractmethod
    async def score_batch_async(
        self,
        goal: str,
        batch: List[Dict[str, str]],
        max_tokens: int,
    ) -> List[float]:
        pass
