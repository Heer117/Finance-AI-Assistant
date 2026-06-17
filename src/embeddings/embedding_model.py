from langchain_huggingface import HuggingFaceEmbeddings
from src.config.settings import settings


def get_embedding_model() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL_NAME)