"""
helpers.py — Shared Utility Functions

WHY: Utility functions are used across chains, UI, and tests.
Centralising them prevents duplication and makes them easy to test.

Includes:
- Input validation
- Text preprocessing
- Response formatting
- Error message generation
"""

from src.config.settings import settings


def validate_text_input(text: str, field_name: str = "Input") -> tuple[bool, str]:
    """
    Validate text input from users.

    Args:
        text: The user-provided text string.
        field_name: Display name for error messages.

    Returns:
        (is_valid: bool, error_message: str)
        If valid, error_message is empty string.
    """
    if not text or not text.strip():
        return False, f"{field_name} cannot be empty."

    if len(text.strip()) < 10:
        return False, f"{field_name} is too short. Please provide more detail."

    if len(text) > settings.MAX_INPUT_LENGTH:
        return (
            False,
            f"{field_name} exceeds maximum length of "
            f"{settings.MAX_INPUT_LENGTH:,} characters. "
            f"Current length: {len(text):,}."
        )

    return True, ""


def truncate_text(text: str, max_chars: int = 500) -> str:
    """
    Truncate text for display purposes.

    Args:
        text: Text to truncate.
        max_chars: Maximum character length.

    Returns:
        Truncated text with ellipsis if necessary.
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."


def format_error_message(error: Exception) -> str:
    """
    Convert an exception into a user-friendly error message.

    Args:
        error: The exception that was raised.

    Returns:
        A clean, actionable error message for display in the UI.
    """
    error_str = str(error).lower()

    # Map common error patterns to friendly messages
    if "api_key" in error_str or "authentication" in error_str or "401" in error_str:
        return (
            "🔑 **API Key Error**: Your Groq API key is invalid or missing. "
            "Please check your `.env` file and ensure `GROQ_API_KEY` is set correctly."
        )

    if "rate_limit" in error_str or "429" in error_str:
        return (
            "⏱️ **Rate Limit Reached**: Too many requests to the Groq API. "
            "Please wait a moment and try again."
        )

    if "timeout" in error_str or "timed out" in error_str:
        return (
            "⌛ **Request Timeout**: The request took too long. "
            "This can happen with very long inputs. Try a shorter text."
        )

    if "connection" in error_str or "network" in error_str:
        return (
            "🌐 **Connection Error**: Could not reach the Groq API. "
            "Please check your internet connection."
        )

    if "model" in error_str and "not found" in error_str:
        return (
            "🤖 **Model Error**: The specified model is not available. "
            "Check your `GROQ_MODEL_NAME` setting in `.env`."
        )

    # Fallback for unexpected errors
    return (
        f"❌ **Unexpected Error**: {str(error)[:200]}. "
        "Please try again or contact support."
    )


def count_words(text: str) -> int:
    """Count words in a string."""
    return len(text.split())


def estimate_tokens(text: str) -> int:
    """
    Rough estimation of token count.
    Rule of thumb: ~4 characters per token for English text.
    """
    return len(text) // 4


def clean_llm_response(response: str) -> str:
    """
    Clean minor formatting artefacts from LLM responses.

    Args:
        response: Raw response string from the LLM.

    Returns:
        Cleaned response string.
    """
    # Remove leading/trailing whitespace
    response = response.strip()

    # Remove any accidental repeated blank lines (max 2 consecutive)
    import re
    response = re.sub(r'\n{3,}', '\n\n', response)

    return response