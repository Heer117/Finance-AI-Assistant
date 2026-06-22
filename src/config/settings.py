"""
settings.py — Centralised Configuration

Works in two environments automatically:
  - Local development: reads from .env file via python-dotenv
  - Streamlit Cloud: reads from st.secrets (set in the dashboard)

The _get() function checks Streamlit secrets first, then falls
back to .env — so the same code works in both places with zero changes.
"""

import os
import streamlit as st
from dotenv import load_dotenv

# Loads .env into environment on local machine
# On Streamlit Cloud this does nothing (no .env file exists there)
load_dotenv()


def _get(key: str, default: str = "") -> str:
    """
    Reads a setting from Streamlit secrets (cloud) first,
    falls back to .env / environment variables (local).
    """
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, default)


class Settings:
    # === LLM ===
    GROQ_API_KEY: str = _get("GROQ_API_KEY")
    GROQ_MODEL_NAME: str = _get("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")

    TEMP_QA: float = 0.1
    TEMP_EXPLAIN: float = 0.3
    TEMP_SUMMARIZE: float = 0.1

    MAX_TOKENS_QA: int = 1024
    MAX_TOKENS_EXPLAIN: int = 1500
    MAX_TOKENS_SUMMARIZE: int = 1024

    GROQ_TIMEOUT: int = 60
    GROQ_MAX_RETRIES: int = 3

    # === App ===
    APP_TITLE: str = "Finance AI Assistant"
    APP_ICON_PATH: str = "assets/trend.png"
    MAX_INPUT_LENGTH: int = 5000

    # === MongoDB / RAG ===
    MONGODB_URI: str = _get("MONGODB_URI")
    MONGODB_DB_NAME: str = _get("MONGODB_DB_NAME", "finance_rag")
    MONGODB_COLLECTION_NAME: str = _get("MONGODB_COLLECTION_NAME", "document_chunks")
    MONGODB_INDEX_NAME: str = _get("MONGODB_INDEX_NAME", "vector_index")

    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    RAG_CHUNK_SIZE: int = 1000
    RAG_CHUNK_OVERLAP: int = 200
    RAG_TOP_K: int = 4

    def validate(self) -> None:
        if not self.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set. "
                "Local: check your .env file. "
                "Cloud: check Streamlit secrets dashboard."
            )

    def validate_mongo(self) -> None:
        if not self.MONGODB_URI:
            raise ValueError(
                "MONGODB_URI is not set. "
                "Local: check your .env file. "
                "Cloud: check Streamlit secrets dashboard."
            )


settings = Settings()