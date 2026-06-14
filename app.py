"""
app.py — Finance AI Assistant (Phase 2 — Fully Stateful)

All three tasks use conversational memory.
Each task has its own independent history stored in session_state.

Finance Q&A      → qa_history      (pure chat interface)
Concept Explain  → explain_history (form + follow-up chat)
Summarization    → summary_history (form + follow-up chat)
"""

import streamlit as st
from PIL import Image
from langchain_core.messages import HumanMessage, AIMessage

from src.config.settings import settings
from src.chains.memory_chain import build_memory_chain
from src.utils.helpers import (
    validate_text_input,
    format_error_message,
    count_words,
    estimate_tokens,
    clean_llm_response,
)

# ── Page config ───────────────────────────────────────────────────────────────
icon = Image.open("assets/trend.png")

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
</style>
""", unsafe_allow_html=True)

# ── Session state — one memory chain, three independent histories ─────────────
if "memory_chain" not in st.session_state:
    st.session_state.memory_chain = build_memory_chain()

# Finance Q&A history
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []
if "qa_display" not in st.session_state:
    st.session_state.qa_display = []

# Concept Explanation history
if "explain_history" not in st.session_state:
    st.session_state.explain_history = []
if "explain_display" not in st.session_state:
    st.session_state.explain_display = []

# Text Summarization history
if "summary_history" not in st.session_state:
    st.session_state.summary_history = []
if "summary_display" not in st.session_state:
    st.session_state.summary_display = []

# ── Helper: append a turn to a history + display store ───────────────────────
def append_turn(history_key, display_key, user_msg, ai_msg):
    st.session_state[history_key].append(HumanMessage(content=user_msg))
    st.session_state[history_key].append(AIMessage(content=ai_msg))
    st.session_state[display_key].append({"role": "user",      "content": user_msg})
    st.session_state[display_key].append({"role": "assistant", "content": ai_msg})

# ── Helper: render stored chat bubbles ───────────────────────────────────────
def render_history(display_key):
    for msg in st.session_state[display_key]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# ── Helper: invoke memory chain ───────────────────────────────────────────────
def ask(history_key, user_message):
    return st.session_state.memory_chain.invoke({
        "history": st.session_state[history_key],
        "question": user_message,
    })

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    col1, col2 = st.columns([2, 15])
    with col1:
        st.image(icon, width=50)
    with col2:
        st.markdown("**Finance AI Assistant**")

    st.divider()

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
    st.markdown("**Model Configuration**")
    st.code(f"Model: {settings.GROQ_MODEL_NAME}\nProvider: Groq", language=None)
    st.divider()

    with st.expander("About this app"):
        st.markdown("""
        **Finance AI Assistant v2.0**
        Phase 2 — Fully Stateful

        All three tasks remember conversation
        history for follow-up questions.

        """)

# ── Main header ───────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 12])
with col1:
    st.image(icon, width=70)
with col2:
    st.title(settings.APP_TITLE)


# ═════════════════════════════════════════════════════════════════════════════
# TASK 1 — Finance Q&A  (pure chat with memory)
# ═════════════════════════════════════════════════════════════════════════════
if task == "Finance Q&A":
    st.header("Finance Question & Answer")
  


    if st.button("New Conversation", key="qa_clear"):
        st.session_state.qa_history = []
        st.session_state.qa_display = []
        st.rerun()

    st.divider()

    # Render existing conversation
    render_history("qa_display")

    # Chat input
    user_input = st.chat_input("Ask a finance question...")

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = ask("qa_history", user_input)
                    response = clean_llm_response(response)
                except Exception as e:
                    st.error(format_error_message(e))
                    st.stop()
            st.markdown(response)

        append_turn("qa_history", "qa_display", user_input, response)


# ═════════════════════════════════════════════════════════════════════════════
# TASK 2 — Concept Explanation  (form + memory for follow-ups)
# ═════════════════════════════════════════════════════════════════════════════
elif task == "Concept Explanation":
    st.header("Financial Concept Explanation")
    st.caption("Conversational — ask follow-ups after the explanation")

    st.markdown("""
    <div class="task-description">
    Enter a concept and get a structured explanation. Then ask follow-up
    questions — change the level, request examples, or go deeper.
    </div>
    """, unsafe_allow_html=True)

    # Form row
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

    if st.button("Clear Explanation History", key="explain_clear"):
        st.session_state.explain_history = []
        st.session_state.explain_display = []
        st.rerun()

    st.divider()

    # When Explain is clicked — construct the prompt and invoke
    if explain_clicked:
        is_valid, error_msg = validate_text_input(concept, "Concept")
        if not is_valid:
            st.error(error_msg)
        else:
            # Craft the user message the same way a user would type it
            user_message = (
                f"Please explain the financial concept '{concept}' "
                f"for a {level} audience."
            )
            with st.spinner(f"Preparing {level} explanation of '{concept}'..."):
                try:
                    response = ask("explain_history", user_message)
                    response = clean_llm_response(response)
                except Exception as e:
                    st.error(format_error_message(e))
                    st.stop()

            append_turn("explain_history", "explain_display", user_message, response)

    # Render conversation so far
    render_history("explain_display")

    # Show follow-up input only if there is at least one exchange
    if st.session_state.explain_history:
        follow_up = st.chat_input(
            "Ask a follow-up — change level, request an example, go deeper..."
        )
        if follow_up:
            with st.chat_message("user"):
                st.markdown(follow_up)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = ask("explain_history", follow_up)
                        response = clean_llm_response(response)
                    except Exception as e:
                        st.error(format_error_message(e))
                        st.stop()
                st.markdown(response)

            append_turn("explain_history", "explain_display", follow_up, response)


# ═════════════════════════════════════════════════════════════════════════════
# TASK 3 — Text Summarization  (form + memory for follow-ups)
# ═════════════════════════════════════════════════════════════════════════════
elif task == "Text Summarization":
    st.header("Financial Text Summarization")
    st.caption("Conversational — ask questions about the summary afterwards")

    st.markdown("""
    <div class="task-description">
    Paste financial text and get a structured summary. Then ask follow-up
    questions about the content — risks, metrics, comparisons, and more.
    </div>
    """, unsafe_allow_html=True)

    with st.expander("See an example input"):
        st.markdown("""
        *Try pasting something like:*
        > Apple Inc. reported quarterly revenue of $94.9 billion, a 6% increase
        > year over year. iPhone revenue was $46.2 billion. Services revenue reached
        > a record $24.2 billion, up 14%. Net income was $23.2 billion.
        """)

    text_input = st.text_area(
        label="Financial Text to Summarize",
        placeholder="Paste your financial article, report, or earnings text here...",
        height=220,
        key="summarize_text",
    )

    if text_input:
        char_count = len(text_input)
        indicator = "🟢" if char_count < 3000 else "🟡" if char_count < 4500 else "🔴"
        st.caption(
            f"{indicator} {char_count:,} / {settings.MAX_INPUT_LENGTH:,} characters"
            f" | ~{estimate_tokens(text_input):,} tokens"
        )

    col1, col2 = st.columns([1, 5])
    with col1:
        summarize_clicked = st.button("Summarize", type="primary", use_container_width=True)
    with col2:
        if st.button("Clear Summary History", key="summary_clear"):
            st.session_state.summary_history = []
            st.session_state.summary_display = []
            st.rerun()

    st.divider()

    # When Summarize is clicked
    if summarize_clicked:
        is_valid, error_msg = validate_text_input(text_input, "Text")
        if not is_valid:
            st.error(error_msg)
        else:
            # Construct a complete message including the text
            user_message = f"Please summarize the following financial text:\n\n{text_input}"

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Original Words", count_words(text_input))

            with st.spinner("Summarizing..."):
                try:
                    response = ask("summary_history", user_message)
                    response = clean_llm_response(response)
                except Exception as e:
                    st.error(format_error_message(e))
                    st.stop()

            with col2:
                st.metric("Summary Words", count_words(response))

            append_turn("summary_history", "summary_display", user_message, response)

    # Render conversation — show only assistant bubbles for the summary itself
    # but full chat for follow-ups (cleaner UX)
    for i, msg in enumerate(st.session_state.summary_display):
        # For the very first user message (the pasted text), show a shorter label
        if msg["role"] == "user" and i == 0:
            with st.chat_message("user"):
                st.markdown("*[Financial text submitted for summarization]*")
        else:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # Show follow-up input only after first summary exists
    if st.session_state.summary_history:
        follow_up = st.chat_input(
            "Ask about the summary — risks, key metrics, comparisons..."
        )
        if follow_up:
            with st.chat_message("user"):
                st.markdown(follow_up)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = ask("summary_history", follow_up)
                        response = clean_llm_response(response)
                    except Exception as e:
                        st.error(format_error_message(e))
                        st.stop()
                st.markdown(response)

            append_turn("summary_history", "summary_display", follow_up, response)

    # Download — only if summary exists
    if st.session_state.summary_display:
        full_convo = "\n\n".join(
            f"{m['role'].upper()}:\n{m['content']}"
            for m in st.session_state.summary_display
        )
        st.download_button(
            label="Download Summary + Follow-ups",
            data=full_convo,
            file_name="financial_summary.txt",
            mime="text/plain",
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
