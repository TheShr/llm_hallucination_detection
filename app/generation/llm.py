import os
from typing import List, Optional

from app.utils.config import settings
from app.utils.logging import logger

try:
    import openai
except ImportError:
    openai = None


class LLMGenerator:
    """LLM generation module with modular provider support."""

    def __init__(self) -> None:
        self.model_name: str = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        if settings.openai_api_key and openai is not None:
            openai.api_key = settings.openai_api_key

    def generate(self, question: str, context: str, max_tokens: int = 250) -> str:
        """Generate an answer given a question and supporting context."""
        prompt = self._build_prompt(question, context)
        logger.info("Calling LLM for generation using model %s", self.model_name)

        if settings.use_openai and openai is not None:
            if self._use_completion_model():
                response = openai.Completion.create(
                    model=self.model_name,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=0.2,
                )
                answer = response["choices"][0]["text"].strip()
            else:
                response = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a grounded assistant that answers based only on provided context."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=0.2,
                )
                answer = response["choices"][0]["message"]["content"].strip()
            return answer

        logger.warning("OpenAI API key not configured; using a fallback template response")
        return f"[Generated answer placeholder]. Context length: {len(context)} characters. Question: {question}"

    def _use_completion_model(self) -> bool:
        """Detect whether the configured model uses the older completion API style."""
        normalized = self.model_name.lower()
        if "grok" in normalized or normalized.startswith("text-") or normalized.startswith("davinci"):
            return True
        return False

    @staticmethod
    def _build_prompt(question: str, context: str) -> str:
        return (
            "Use the context below to answer the question. If the answer is not in the context, say you cannot answer it."
            f"\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
