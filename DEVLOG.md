# Development Log

Daily progress notes for the Finance AI Assistant project, built as part of a summer internship.

---

## Day 1

**Goal:** Environment setup and architecture understanding

- Set up Python virtual environment
- Resolved a Python version compatibility issue — Python 3.14 had no pre-built numpy wheel,
  causing a Meson/GCC build failure. Fixed by switching to Python 3.11.
- Installed and verified all dependencies (LangChain, langchain-groq, Streamlit, etc.)
- Studied LangChain fundamentals: ChatPromptTemplate, LCEL pipe syntax, ChatGroq, StrOutputParser
- Planned project folder structure following separation of concerns

---

## Day 2

**Goal:** Phase 1 — working stateless application

- Implemented `settings.py`, `groq_client.py`, `finance_prompts.py`, and chain definitions
- Built Streamlit UI with three tasks: Finance Q&A, Concept Explanation, Text Summarization
- Tested all three features end-to-end with the Groq API
- Phase 1 fully functional

---

## Day 3

**Goal:** Phase 2 — conversational memory

- Learned LangChain's LCEL-native memory pattern using `MessagesPlaceholder`
  (the modern alternative to legacy `ConversationBufferMemory`)
- Built a unified memory chain that handles Q&A, explanation, and summarization
  within a single conversation
- Converted all three tasks to stateful chat interfaces using `st.session_state`
- Verified follow-up questions correctly use prior conversation context
- Set up GitHub repository with README, .gitignore, and project documentation

---

## Key Learnings So Far

- LCEL's `prompt | llm | parser` pattern makes chains modular and provider-agnostic
- Memory in LangChain 0.3 is implemented by manually replaying message history,
  not by a "memory module" that does anything magical
- Temperature should be tuned per task — low (0.1) for factual/summarization tasks,
  higher (0.3+) for explanation tasks needing analogies
- Clean separation between UI, prompts, chains, and config makes it possible to
  add new capabilities (RAG, agents) without touching existing code
