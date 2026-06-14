"""
groq_client.py — LLM Client for Groq API

Creates configured ChatGroq instances per task type.
Each task gets its own temperature setting:
  - QA         → 0.1  (factual, deterministic)
  - Explanation → 0.3  (slightly creative for analogies)
  - Summarize   → 0.1  (faithful to source text)
"""

from langchain_groq import ChatGroq
from src.config.settings import settings


def create_llm(temperature: float = 0.1, max_tokens: int = 1024) -> ChatGroq:
    """Factory function — creates a configured ChatGroq instance."""
    settings.validate()
    return ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model=settings.GROQ_MODEL_NAME,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=settings.GROQ_TIMEOUT,
        max_retries=settings.GROQ_MAX_RETRIES,
    )


def get_qa_llm() -> ChatGroq:
    """Low temperature — factual Q&A."""
    return create_llm(temperature=settings.TEMP_QA, max_tokens=settings.MAX_TOKENS_QA)


def get_explain_llm() -> ChatGroq:
    """Slightly higher temperature — better analogies in explanations."""
    return create_llm(temperature=settings.TEMP_EXPLAIN, max_tokens=settings.MAX_TOKENS_EXPLAIN)


def get_summarize_llm() -> ChatGroq:
    """Low temperature — stay faithful to the source text."""
    return create_llm(temperature=settings.TEMP_SUMMARIZE, max_tokens=settings.MAX_TOKENS_SUMMARIZE)