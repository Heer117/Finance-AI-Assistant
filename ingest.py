from langchain_mongodb import (
    MongoDBAtlasVectorSearch
)

from embeddings.embedding_model import (
    get_embeddings
)

vectorstore = MongoDBAtlasVectorSearch(
    collection=collection,
    embedding=get_embeddings(),
    index_name="vector_index"
)

vectorstore.add_documents(chunks)