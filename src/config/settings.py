import os
from dotenv import load_dotenv

load_dotenv()

class Settings:

    # === LLM Configuration ===
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL_NAME: str = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")

    # Temperature per task type — lower = more deterministic
    TEMP_QA: float = 0.1          # Question Answering
    TEMP_EXPLAIN: float = 0.3     # Concept Explanation
    TEMP_SUMMARIZE: float = 0.1   # Summarization (stay faithful to text)
   

    # Token limits per task
    MAX_TOKENS_QA: int = 1024
    MAX_TOKENS_EXPLAIN: int = 1500
    MAX_TOKENS_SUMMARIZE: int = 1024
   

    # === API Configuration ===
    GROQ_TIMEOUT: int = 60        # Request timeout in seconds
    GROQ_MAX_RETRIES: int = 3     # Number of retries on failure

    # === App Configuration ===
    APP_TITLE: str = "Finance AI Assistant"
    APP_ICON: str = "assets/trend.png"
    MAX_INPUT_LENGTH: int = 5000  # Max characters in user text input

    def validate(self) -> None:
        """Raise an error if required settings are missing."""
        if not self.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set. "
                "Please add it to your .env file: GROQ_API_KEY=gsk_..."
            )
        if len(self.GROQ_API_KEY) < 10:
            raise ValueError("GROQ_API_KEY appears to be invalid (too short).")


# Create a singleton instance — import this everywhere
settings = Settings()