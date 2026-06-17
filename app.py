"""
app.py — Finance AI Assistant (Phase 3 — RAG Integrated into All Tasks)

HOW RAG IS INTEGRATED:
  The sidebar has a persistent PDF upload section.
  When a document is ingested, ALL three tasks automatically switch
  from the Phase 2 memory chain to the Phase 3 RAG-enhanced chain.
  Every response is now grounded in the uploaded document's content,
  and a "Document Sources" expander shows which chunks were used.

  No document uploaded → Phase 2 behaviour (general LLM knowledge)
  Document uploaded    → Phase 3 behaviour (document-grounded answers)

Tasks (unchanged in UI, upgraded in capability):
  1. Finance Q&A          — now answers from the document when relevant
  2. Concept Explanation  — now references document examples
  3. Text Summarization   — now uses document context to enrich summaries
"""

import tempfile
import os as os_module

import streamlit as st
from PIL import Image
from langchain_core.messages import HumanMessage, AIMessage

from src.config.settings import settings
from src.chains.memory_chain import (
    build_memory_chain,
    build_rag_enhanced_chain,
    format_docs,
)
from src.ingestion.pdf_processor import load_and_chunk_pdf
from src.vectorstore.mongo_vectorstore import (
    add_documents_to_store,
    clear_collection,
)
from src.utils.helpers import (
    validate_text_input,
    format_error_message,
    count_words,
    estimate_tokens,
    clean_llm_response,
)

# ── Page config ───────────────────────────────────────────────────────────────
icon = Image.open(settings.APP_ICON_PATH)

st.set_page_config(
    page_title=settings.APP_TITLE,
    page_icon=icon,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 900px;
    }
    .task-description {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin-bottom: 1.5rem;
        color: #e2e8f0;
        font-size: 0.9rem;
    }
    .doc-active-banner {
        background: linear-gradient(135deg, #064e3b 0%, #065f46 100%);
        border: 1px solid #10b981;
        border-radius: 12px;
        padding: 0.75rem 1rem;
        margin-bottom: 1rem;
        color: #d1fae5;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────

# Conversation histories (per task)
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []
if "qa_display" not in st.session_state:
    st.session_state.qa_display = []

if "explain_history" not in st.session_state:
    st.session_state.explain_history = []
if "explain_display" not in st.session_state:
    st.session_state.explain_display = []

if "summary_history" not in st.session_state:
    st.session_state.summary_history = []
if "summary_display" not in st.session_state:
    st.session_state.summary_display = []

# Document / RAG state
if "document_ingested" not in st.session_state:
    st.session_state.document_ingested = False
if "document_info" not in st.session_state:
    st.session_state.document_info = {}

# Chain — rebuilt whenever document status changes
if "active_chain" not in st.session_state:
    st.session_state.active_chain = build_memory_chain()
if "active_retriever" not in st.session_state:
    st.session_state.active_retriever = None


# ── Chain selection helper ────────────────────────────────────────────────────
def get_active_chain():
    """
    Returns the correct chain based on whether a document is ingested.
    Phase 2 (no doc) → memory chain
    Phase 3 (doc)    → RAG-enhanced chain
    """
    return st.session_state.active_chain


# ── Shared helpers ────────────────────────────────────────────────────────────

def append_turn(history_key, display_key, user_msg, ai_msg, sources=None):
    st.session_state[history_key].append(HumanMessage(content=user_msg))
    st.session_state[history_key].append(AIMessage(content=ai_msg))
    entry = {"role": "user",      "content": user_msg}
    st.session_state[display_key].append(entry)
    response_entry = {"role": "assistant", "content": ai_msg}
    if sources:
        response_entry["sources"] = sources
    st.session_state[display_key].append(response_entry)


def render_history(display_key):
    for msg in st.session_state[display_key]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg:
                with st.expander("Document Sources"):
                    st.markdown(msg["sources"])


def ask(history_key, user_message):
    """Invokes the active chain and returns (response, sources_text)."""
    chain = get_active_chain()

    response = chain.invoke({
        "history": st.session_state[history_key],
        "question": user_message,
    })
    response = clean_llm_response(response)

    # Retrieve sources for display if RAG is active
    sources_text = None
    if st.session_state.document_ingested and st.session_state.active_retriever:
        retrieved_docs = st.session_state.active_retriever.invoke(user_message)
        sources_text = format_docs(retrieved_docs)

    return response, sources_text


def show_doc_banner():
    """Shows a small green banner when a document is active."""
    if st.session_state.document_ingested:
        info = st.session_state.document_info
        st.markdown(
            f"""<div class="doc-active-banner">
            Document active: <strong>{info.get('filename', '')}</strong>
            ({info.get('pages', '?')} pages, {info.get('chunks', '?')} chunks) —
            answers are grounded in this document.
            </div>""",
            unsafe_allow_html=True,
        )


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    col1, col2 = st.columns([2, 15])
    with col1:
        st.image(icon, width=50)
    with col2:
        st.markdown("**Finance AI Assistant**")

    st.divider()

    # ── PDF Upload (Phase 3) — always visible in sidebar ─────────────────────
    st.markdown("**Document (RAG)**")
    st.caption("Upload a PDF to ground all tasks in its content.")

    uploaded_file = st.file_uploader(
        label="Upload PDF",
        type=["pdf"],
        label_visibility="collapsed",
    )

    col1, col2 = st.columns(2)
    with col1:
        ingest_clicked = st.button(
            "Process",
            use_container_width=True,
            type="primary",
            disabled=(uploaded_file is None),
        )
    with col2:
        clear_clicked = st.button(
            "Clear Doc",
            use_container_width=True,
            disabled=(not st.session_state.document_ingested),
        )

    # Ingestion
    if ingest_clicked and uploaded_file is not None:
        with st.spinner("Processing PDF..."):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                chunks = load_and_chunk_pdf(tmp_path)
                clear_collection()
                num_stored = add_documents_to_store(chunks)
                os_module.unlink(tmp_path)

                pages = max(
                    (c.metadata.get("page", 0) for c in chunks), default=0
                ) + 1

                # Switch to RAG chain
                chain, retriever = build_rag_enhanced_chain()
                st.session_state.active_chain = chain
                st.session_state.active_retriever = retriever
                st.session_state.document_ingested = True
                st.session_state.document_info = {
                    "filename": uploaded_file.name,
                    "pages": pages,
                    "chunks": num_stored,
                }

                # Clear all conversation histories
                # (previous conversations used general knowledge;
                #  new conversations will use document context)
                for key in ["qa_history", "qa_display",
                            "explain_history", "explain_display",
                            "summary_history", "summary_display"]:
                    st.session_state[key] = []

                st.rerun()

            except Exception as e:
                st.error(format_error_message(e))

    # Clear document
    if clear_clicked:
        try:
            clear_collection()
        except Exception:
            pass  # ignore mongo errors on clear
        st.session_state.active_chain = build_memory_chain()
        st.session_state.active_retriever = None
        st.session_state.document_ingested = False
        st.session_state.document_info = {}
        for key in ["qa_history", "qa_display",
                    "explain_history", "explain_display",
                    "summary_history", "summary_display"]:
            st.session_state[key] = []
        st.rerun()

    # Document status
    if st.session_state.document_ingested:
        info = st.session_state.document_info
        st.success("Document active")
        st.caption(f"{info.get('filename', '')}")
        st.caption(f"{info.get('pages', '?')} pages · {info.get('chunks', '?')} chunks")
    else:
        st.info("No document — using general knowledge")

    st.divider()

    # Task selector
    st.markdown("**SELECT TASK**")
    task = st.selectbox(
        label="Task",
        options=[
            "Finance Q&A",
            "Concept Explanation",
            "Text Summarization",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("**Model**")
    st.code(
        f"Model: {settings.GROQ_MODEL_NAME}\n"
        f"Provider: Groq\n"
        f"RAG: {'Active' if st.session_state.document_ingested else 'Off'}",
        language=None,
    )

    st.divider()
    with st.expander("About this app"):
        st.markdown("""
**Finance AI Assistant v3.0**
Phase 3 — RAG Integrated

Upload a PDF to activate document-grounded
answers across all three tasks.

Without document: general LLM knowledge
With document: answers from your PDF

Built with LangChain · Groq · MongoDB Atlas · Streamlit
        """)


# ── Main header ───────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 12])
with col1:
    st.image(icon, width=70)
with col2:
    st.title(settings.APP_TITLE)


# ═════════════════════════════════════════════════════════════════════════════
# TASK 1 — Finance Q&A
# ═════════════════════════════════════════════════════════════════════════════
if task == "Finance Q&A":
    st.header("Finance Question & Answer")
    st.caption(
        "Phase 3 — Document-grounded"
        if st.session_state.document_ingested
        else "Phase 2 — Conversational memory"
    )

    show_doc_banner()

    st.markdown("""
    <div class="task-description">
    Ask any finance question. The assistant remembers context across messages.
    If a document is uploaded, answers are grounded in its content.
    </div>
    """, unsafe_allow_html=True)

    if st.button("New Conversation", key="qa_clear"):
        st.session_state.qa_history = []
        st.session_state.qa_display = []
        st.rerun()

    st.divider()

    render_history("qa_display")

    user_input = st.chat_input("Ask a finance question...")

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response, sources = ask("qa_history", user_input)
                except Exception as e:
                    st.error(format_error_message(e))
                    st.stop()
            st.markdown(response)
            if sources:
                with st.expander("Document Sources"):
                    st.markdown(sources)

        append_turn("qa_history", "qa_display", user_input, response, sources)


# ═════════════════════════════════════════════════════════════════════════════
# TASK 2 — Concept Explanation
# ═════════════════════════════════════════════════════════════════════════════
elif task == "Concept Explanation":
    st.header("Financial Concept Explanation")
    st.caption(
        "Phase 3 — Document-grounded"
        if st.session_state.document_ingested
        else "Phase 2 — Conversational memory"
    )

    show_doc_banner()

    st.markdown("""
    <div class="task-description">
    Get a structured explanation of any financial concept. Choose your level.
    If a document is uploaded, the explanation references examples from it.
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([4, 2, 1])
    with col1:
        concept = st.text_input(
            label="Financial Concept",
            placeholder="e.g. Yield Curve, Working Capital, DCF Valuation",
            key="concept_input",
        )
    with col2:
        level = st.selectbox(
            label="Level",
            options=["beginner", "intermediate", "advanced"],
            index=1,
            key="concept_level",
        )
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        explain_clicked = st.button("Explain", type="primary", use_container_width=True)

    if st.button("Clear History", key="explain_clear"):
        st.session_state.explain_history = []
        st.session_state.explain_display = []
        st.rerun()

    st.divider()

    if explain_clicked:
        is_valid, error_msg = validate_text_input(concept, "Concept")
        if not is_valid:
            st.error(error_msg)
        else:
            user_message = (
                f"Please explain the financial concept '{concept}' "
                f"for a {level} audience."
                + (
                    " Reference the uploaded document if it contains "
                    "relevant examples or data."
                    if st.session_state.document_ingested else ""
                )
            )
            with st.spinner(f"Preparing {level} explanation of '{concept}'..."):
                try:
                    response, sources = ask("explain_history", user_message)
                except Exception as e:
                    st.error(format_error_message(e))
                    st.stop()

            append_turn("explain_history", "explain_display",
                        user_message, response, sources)

    render_history("explain_display")

    if st.session_state.explain_history:
        follow_up = st.chat_input("Ask a follow-up...")
        if follow_up:
            with st.chat_message("user"):
                st.markdown(follow_up)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response, sources = ask("explain_history", follow_up)
                    except Exception as e:
                        st.error(format_error_message(e))
                        st.stop()
                st.markdown(response)
                if sources:
                    with st.expander("Document Sources"):
                        st.markdown(sources)

            append_turn("explain_history", "explain_display",
                        follow_up, response, sources)


# ═════════════════════════════════════════════════════════════════════════════
# TASK 3 — Text Summarization
# ═════════════════════════════════════════════════════════════════════════════
elif task == "Text Summarization":
    st.header("Financial Text Summarization")
    st.caption(
        "Phase 3 — Document-grounded"
        if st.session_state.document_ingested
        else "Phase 2 — Conversational memory"
    )

    show_doc_banner()

    st.markdown("""
    <div class="task-description">
    Paste financial text and get a structured summary. Ask follow-up questions
    about the content. If a document is uploaded, the assistant also draws
    on it to enrich the summary with broader context.
    </div>
    """, unsafe_allow_html=True)

    with st.expander("See an example input"):
        st.markdown("""
        > Apple Inc. reported quarterly revenue of $94.9 billion, a 6% increase
        > year over year. iPhone revenue was $46.2 billion. Services revenue reached
        > a record $24.2 billion, up 14%. Net income was $23.2 billion.
        """)

    text_input = st.text_area(
        label="Financial Text to Summarize",
        placeholder="Paste your financial article, report excerpt, or earnings text here...",
        height=200,
        key="summarize_text",
    )

    if text_input:
        char_count = len(text_input)
        indicator = "🟢" if char_count < 3000 else "🟡" if char_count < 4500 else "🔴"
        st.caption(
            f"{indicator} {char_count:,} / {settings.MAX_INPUT_LENGTH:,} characters"
        )

    col1, col2 = st.columns([1, 5])
    with col1:
        summarize_clicked = st.button(
            "Summarize", type="primary", use_container_width=True
        )
    with col2:
        if st.button("Clear History", key="summary_clear"):
            st.session_state.summary_history = []
            st.session_state.summary_display = []
            st.rerun()

    st.divider()

    if summarize_clicked:
        is_valid, error_msg = validate_text_input(text_input, "Text")
        if not is_valid:
            st.error(error_msg)
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Original Words", count_words(text_input))

            user_message = (
                f"Please summarize the following financial text:\n\n{text_input}"
                + (
                    "\n\nAlso reference the uploaded document if it provides "
                    "relevant context or related data."
                    if st.session_state.document_ingested else ""
                )
            )

            with st.spinner("Summarizing..."):
                try:
                    response, sources = ask("summary_history", user_message)
                except Exception as e:
                    st.error(format_error_message(e))
                    st.stop()

            with col2:
                st.metric("Summary Words", count_words(response))

            append_turn("summary_history", "summary_display",
                        user_message, response, sources)

    # Render — replace long pasted text with a placeholder label
    for i, msg in enumerate(st.session_state.summary_display):
        if msg["role"] == "user" and i == 0:
            with st.chat_message("user"):
                st.markdown("*[Financial text submitted for summarization]*")
        else:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "sources" in msg:
                    with st.expander("Document Sources"):
                        st.markdown(msg["sources"])

    if st.session_state.summary_history:
        follow_up = st.chat_input("Ask about the summary...")
        if follow_up:
            with st.chat_message("user"):
                st.markdown(follow_up)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response, sources = ask("summary_history", follow_up)
                    except Exception as e:
                        st.error(format_error_message(e))
                        st.stop()
                st.markdown(response)
                if sources:
                    with st.expander("Document Sources"):
                        st.markdown(sources)

            append_turn("summary_history", "summary_display",
                        follow_up, response, sources)

        if st.session_state.summary_display:
            full_convo = "\n\n".join(
                f"{m['role'].upper()}:\n{m['content']}"
                for m in st.session_state.summary_display
            )
            st.download_button(
                label="Download Summary",
                data=full_convo,
                file_name="financial_summary.txt",
                mime="text/plain",
            )


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    """
    <div style="text-align:center; color:#94a3b8; font-size:0.8rem;">
    <strong>Disclaimer:</strong> For educational purposes only. Not financial advice.
    | Finance AI Assistant v3.0 — Phase 3
    </div>
    """,
    unsafe_allow_html=True,
)