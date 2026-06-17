"""
settings.py — Centralised Configuration (Phase 1 + 2 + 3)

Phase 3 adds MongoDB Atlas and embedding/chunking configuration
for the RAG pipeline.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # === LLM (Phase 1) ===
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL_NAME: str = os.getenv("GROQ_MODEL_NAME", "llama-3.1-70b-versatile")

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

    # === Phase 3: RAG / MongoDB Configuration ===
    MONGODB_URI: str = os.getenv("MONGODB_URI", "mongodb+srv://finance_user:vdHeer3010@finance-rag-cluster.xlmnycq.mongodb.net/?appName=finance-rag-cluster")
    MONGODB_DB_NAME: str = os.getenv("MONGODB_DB_NAME", "finance_rag")
    MONGODB_COLLECTION_NAME: str = os.getenv("MONGODB_COLLECTION_NAME", "document_chunks")
    MONGODB_INDEX_NAME: str = os.getenv("MONGODB_INDEX_NAME", "vector_index")

    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    RAG_CHUNK_SIZE: int = 1000
    RAG_CHUNK_OVERLAP: int = 200
    RAG_TOP_K: int = 4

    def validate(self) -> None:
        """Raise an error if required Groq settings are missing."""
        if not self.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set. Please add it to your .env file."
            )

    def validate_mongo(self) -> None:
        """Raise an error if required MongoDB settings are missing."""
        if not self.MONGODB_URI:
            raise ValueError(
                "MONGODB_URI is not set. Add it to your .env file. "
                "See Phase 3 guide section 5 for setup instructions."
            )


settings = Settings()
