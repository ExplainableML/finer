# ash_eval/models/base.py

from abc import ABC, abstractmethod


class VisionLanguageModelAdapter(ABC):
    @abstractmethod
    def generate_text(self, messages, max_new_tokens: int) -> str:
        """
        Input:
            messages: OpenAI-style multimodal chat messages

        Output:
            raw generated text
        """
        raise NotImplementedError