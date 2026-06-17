"""
memory_chain.py — Conversational Chains (Phase 2 + Phase 3)

Two chains are defined here:

  build_memory_chain()
    Phase 2 — no document context.
    Used when no PDF has been uploaded.
    Input: {"history": [...], "question": str}

  build_rag_enhanced_chain()
    Phase 3 — retrieves relevant document chunks before answering.
    Used across ALL tasks when a PDF has been ingested into MongoDB.
    Input: {"history": [...], "question": str}
    Internally: retrieves top-K chunks from MongoDB, injects as context.

HOW RAG IS INTEGRATED INTO ALL TASKS:
  Every task (Q&A, Explanation, Summarization) sends a question string.
  If a document is ingested, the RAG chain retrieves relevant chunks
  from that document before answering — grounding the response in
  the uploaded file rather than the LLM's general knowledge alone.

  app.py checks st.session_state.document_ingested and picks
  the right chain automatically. No changes needed to the task UIs.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from src.llm.groq_client import get_qa_llm
from src.vectorstore.mongo_vectorstore import get_retriever


# ─── Shared Finance Persona ───────────────────────────────────────────────────

UNIFIED_FINANCE_PERSONA = """You are FinanceAI, a Senior Financial Analyst and Educator \
with 20 years of experience across investment banking, corporate finance, equity research, \
and financial planning.

You handle three types of requests in a single conversation:

1. FINANCE Q&A — When the user asks a finance question, provide:
   Direct Answer, Detailed Explanation, Key Points, and an Example.

2. CONCEPT EXPLANATION — When the user asks to explain a concept, provide:
   Definition, Simple analogy, How it works, Why it matters, Related concepts.
   Default to intermediate level unless the user specifies otherwise.

3. TEXT SUMMARIZATION — When the user pastes text and asks for a summary, provide:
   Executive Summary, Key Metrics, Main Topics, Critical Findings, Risks.

MEMORY: You remember everything in this conversation. Use prior context
for all follow-up questions. Never ask the user to repeat themselves.

CONSTRAINTS:
- Only respond to finance, economics, accounting, and investment topics
- Never recommend specific stocks or investment products
- Decline non-financial topics politely"""


UNIFIED_FINANCE_PERSONA_WITH_RAG = UNIFIED_FINANCE_PERSONA + """

DOCUMENT CONTEXT:
You also have access to an uploaded financial document (annual report,
financial statement, or similar). Relevant excerpts from this document
will be provided with each question.

When document excerpts are provided:
- Prioritise information from the document over general knowledge
- Reference specific figures, sections, or statements from the document
- If the document does not contain relevant information for the question,
  say so and answer from general financial knowledge instead
- Always make it clear when your answer is based on the uploaded document"""


# ─── Helper: format retrieved docs into a readable context string ─────────────

def format_docs(docs) -> str:
    """
    Joins retrieved document chunks into a single string,
    labeling each with its source page for traceability.
    Used both inside the chain and in app.py for the Sources display.
    """
    if not docs:
        return "No relevant document excerpts found."
    parts = []
    for doc in docs:
        page = doc.metadata.get("page", "unknown")
        source = doc.metadata.get("source", "document")
        parts.append(f"[{source} — Page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


# ─── Phase 2: Memory Chain (no document context) ──────────────────────────────

def build_memory_chain():
    """
    Phase 2 chain — conversational memory, no document retrieval.
    Used when no PDF has been uploaded.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", UNIFIED_FINANCE_PERSONA),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])

    return prompt | get_qa_llm() | StrOutputParser()


# ─── Phase 3: RAG-Enhanced Chain (with document context) ─────────────────────

def build_rag_enhanced_chain():
    """
    Phase 3 chain — retrieves relevant document chunks before every answer.
    Used across ALL tasks when a document has been ingested into MongoDB.

    Returns:
        chain     — invoke with {"history": [...], "question": str}
        retriever — used separately in app.py to display source chunks
    """
    retriever = get_retriever()

    prompt = ChatPromptTemplate.from_messages([
        ("system", UNIFIED_FINANCE_PERSONA_WITH_RAG),
        MessagesPlaceholder(variable_name="history"),
        ("human", """Relevant excerpts from the uploaded document:
{context}

Question: {question}"""),
    ])

    def retrieve_context(inputs: dict) -> str:
        """Retrieves and formats relevant chunks for the question."""
        docs = retriever.invoke(inputs["question"])
        return format_docs(docs)

    chain = (
        {
            "context":  RunnableLambda(retrieve_context),
            "history":  RunnableLambda(lambda x: x["history"]),
            "question": RunnableLambda(lambda x: x["question"]),
        }
        | prompt
        | get_qa_llm()
        | StrOutputParser()
    )

    return chain, retriever