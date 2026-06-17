from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from src.config.settings import settings
from src.embeddings.embedding_model import get_embedding_model


def get_mongo_collection():
    """Returns the MongoDB collection used for storing document chunks."""
    settings.validate_mongo()
    client = MongoClient(settings.MONGODB_URI)
    db = client[settings.MONGODB_DB_NAME]
    return db[settings.MONGODB_COLLECTION_NAME]


def get_vector_store() -> MongoDBAtlasVectorSearch:
    """Returns a LangChain-compatible MongoDB Atlas vector store."""
    collection = get_mongo_collection()
    embeddings = get_embedding_model()

    return MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name=settings.MONGODB_INDEX_NAME,
        text_key="text",
        embedding_key="embedding",
    )


def add_documents_to_store(chunks) -> int:
    """Embeds and stores document chunks in MongoDB. Returns chunk count."""
    vector_store = get_vector_store()
    vector_store.add_documents(chunks)
    return len(chunks)


def get_retriever(k: int = None):
    """Returns a retriever for similarity search over stored chunks."""
    k = k or settings.RAG_TOP_K
    vector_store = get_vector_store()
    return vector_store.as_retriever(search_kwargs={"k": k})


def clear_collection():
    """Removes all chunks — useful for re-uploading a fresh document."""
    collection = get_mongo_collection()
    collection.delete_many({})