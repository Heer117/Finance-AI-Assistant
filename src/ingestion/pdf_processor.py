import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config.settings import settings


def clean_text(text: str) -> str:
    """Light cleanup of PDF extraction artifacts."""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def load_and_chunk_pdf(file_path: str):
    """
    Loads a PDF and splits it into chunks ready for embedding.

    Returns:
        list[Document] — each chunk with metadata {source, page}
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Clean text and tag with source filename
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)
        doc.metadata["source"] = file_path.split("/")[-1].split("\\")[-1]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.RAG_CHUNK_SIZE,
        chunk_overlap=settings.RAG_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)
    return chunks