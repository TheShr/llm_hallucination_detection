import os
from typing import Optional

import requests

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
            if settings.openai_api_base:
                if hasattr(openai, "base_url"):
                    openai.base_url = settings.openai_api_base
                else:
                    openai.api_base = settings.openai_api_base

    def generate(self, question: str, context: str, max_tokens: int = 250) -> str:
        """Generate an answer given a question and supporting context."""
        prompt = self._build_prompt(question, context)
        logger.info("Calling LLM for generation using model %s", self.model_name)

        if settings.use_openai and openai is not None:
            try:
                # First try the ChatCompletion endpoint, which remains supported by many OpenAI-compatible backends.
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
            except AttributeError:
                logger.warning("ChatCompletion not supported; trying the Responses API instead.")
            except Exception as exc:
                logger.warning("ChatCompletion failed: %s", exc)

            try:
                result = self._try_responses_api(prompt, max_tokens)
                if result:
                    return result
            except Exception as exc:
                logger.warning("Responses fallback failed: %s", exc)

        logger.warning("OpenAI API key not configured or OpenAI generation failed; using a fallback error response")
        return (
            "[LLM backend unavailable] Unable to generate an answer because the OpenAI/Grok API is not configured or failed. "
            "Please set OPENAI_API_KEY and verify connectivity to your model provider."
        )

    def _try_responses_api(self, prompt: str, max_tokens: int) -> Optional[str]:
        if openai is not None and hasattr(openai, "responses"):
            logger.info("Trying openai.responses.create fallback")
            response = openai.responses.create(
                model=self.model_name,
                input=prompt,
                max_tokens=max_tokens,
                temperature=0.2,
            )
            if hasattr(response, "output_text"):
                return response.output_text.strip()
            output = response.get("output")
            if isinstance(output, list) and output:
                first = output[0]
                if isinstance(first, dict) and "content" in first:
                    return first["content"][0]["text"].strip()
            if isinstance(response, dict):
                texts = []
                for item in response.get("output", []):
                    if isinstance(item, dict) and "content" in item:
                        for chunk in item["content"]:
                            if isinstance(chunk, dict) and "text" in chunk:
                                texts.append(chunk["text"])
                if texts:
                    return "".join(texts).strip()

        if settings.openai_api_base and settings.openai_api_key:
            logger.info("Trying direct HTTP responses fallback to %s", settings.openai_api_base)
            return self._call_direct_responses(prompt, max_tokens)

        return None

    def _call_direct_responses(self, prompt: str, max_tokens: int) -> Optional[str]:
        base_url = settings.openai_api_base.rstrip("/")
        url = f"{base_url}/responses"
        headers = {
            "Authorization": f"Bearer {settings.openai_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "input": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        body = response.json()
        if isinstance(body, dict) and "output_text" in body:
            return body["output_text"].strip()
        if isinstance(body, dict) and "output" in body:
            output = body["output"]
            if isinstance(output, list) and output:
                first = output[0]
                if isinstance(first, dict) and "content" in first:
                    if isinstance(first["content"], list):
                        texts = []
                        for chunk in first["content"]:
                            if isinstance(chunk, dict) and "text" in chunk:
                                texts.append(chunk["text"])
                        if texts:
                            return "".join(texts).strip()
            if isinstance(output, str):
                return output.strip()
        return None

    @staticmethod
    def _build_prompt(question: str, context: str) -> str:
        return (
            "Use the context below to answer the question. If the answer is not in the context, say you cannot answer it."
            f"\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
