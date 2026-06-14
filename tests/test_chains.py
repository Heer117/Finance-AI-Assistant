"""
test_chains.py — Unit Tests for LangChain Chains

TESTING STRATEGY for Phase 1:
  - Unit tests: Test prompt building (no API calls)
  - Integration tests: Test full chain with real API (optional, rate-limited)
  - Mock tests: Test chain logic with mocked LLM responses

WHY TEST PROMPTS?
  Prompts are logic. A broken prompt produces wrong output.
  Test that prompts format inputs correctly before API costs are incurred.
"""

import pytest
from langchain_core.prompts import ChatPromptTemplate
from src.prompts.finance_prompts import (
    get_qa_prompt,
    get_explanation_prompt,
    get_summarization_prompt,
    get_report_prompt,
    get_insights_prompt,
)
from src.utils.helpers import validate_text_input, format_error_message


class TestPromptTemplates:
    """Test that prompt templates format inputs correctly."""

    def test_qa_prompt_has_correct_input_variable(self):
        """QA prompt should accept 'question' as its only variable."""
        prompt = get_qa_prompt()
        assert "question" in prompt.input_variables

    def test_qa_prompt_formats_correctly(self):
        """Formatted QA prompt should contain the user's question."""
        prompt = get_qa_prompt()
        messages = prompt.format_messages(question="What is EBITDA?")
        # The human message should contain the question
        human_message = messages[-1]
        assert "What is EBITDA?" in human_message.content

    def test_explanation_prompt_variables(self):
        """Explanation prompt needs concept and level variables."""
        prompt = get_explanation_prompt()
        assert "concept" in prompt.input_variables
        assert "level" in prompt.input_variables

    def test_explanation_prompt_formats_level(self):
        """Level should appear in the formatted prompt."""
        prompt = get_explanation_prompt()
        messages = prompt.format_messages(
            concept="Yield Curve", level="beginner"
        )
        combined = " ".join(m.content for m in messages)
        assert "beginner" in combined
        assert "Yield Curve" in combined

    def test_summarization_prompt_variables(self):
        """Summarization prompt needs 'text' variable."""
        prompt = get_summarization_prompt()
        assert "text" in prompt.input_variables

    def test_report_prompt_has_all_variables(self):
        """Report prompt needs 4 specific variables."""
        prompt = get_report_prompt()
        for var in ["company_name", "report_type", "time_period", "data_points"]:
            assert var in prompt.input_variables

    def test_insights_prompt_variables(self):
        """Insights prompt needs text and focus_area."""
        prompt = get_insights_prompt()
        assert "text" in prompt.input_variables
        assert "focus_area" in prompt.input_variables


class TestInputValidation:
    """Test the input validation utility functions."""

    def test_empty_string_is_invalid(self):
        is_valid, msg = validate_text_input("")
        assert not is_valid
        assert "empty" in msg.lower()

    def test_whitespace_only_is_invalid(self):
        is_valid, msg = validate_text_input("   ")
        assert not is_valid

    def test_very_short_text_is_invalid(self):
        is_valid, msg = validate_text_input("Hi")
        assert not is_valid
        assert "short" in msg.lower()

    def test_valid_text_passes(self):
        is_valid, msg = validate_text_input("What is the price-to-earnings ratio?")
        assert is_valid
        assert msg == ""

    def test_text_over_max_length_is_invalid(self):
        long_text = "x" * 6000  # Over MAX_INPUT_LENGTH of 5000
        is_valid, msg = validate_text_input(long_text)
        assert not is_valid
        assert "exceeds" in msg.lower()


class TestErrorMessages:
    """Test error message formatting."""

    def test_api_key_error_gives_helpful_message(self):
        error = Exception("authentication error: invalid api_key")
        msg = format_error_message(error)
        assert "API Key" in msg
        assert ".env" in msg

    def test_rate_limit_error_gives_helpful_message(self):
        error = Exception("rate_limit exceeded 429")
        msg = format_error_message(error)
        assert "Rate Limit" in msg

    def test_timeout_error_gives_helpful_message(self):
        error = Exception("request timed out after 60s")
        msg = format_error_message(error)
        assert "Timeout" in msg

    def test_unknown_error_returns_fallback(self):
        error = Exception("some weird unknown error happened")
        msg = format_error_message(error)
        assert "Unexpected Error" in msg


# ─── Run tests ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pytest.main([__file__, "-v"])