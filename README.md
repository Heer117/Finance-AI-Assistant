# Finance AI Assistant

A conversational AI assistant for finance-related question answering, concept explanation, and financial text summarization — built with LangChain, Groq, and Streamlit.

This is an evolving project. Currently implements **Phase 1 (stateless task execution)** and **Phase 2 (conversational memory)**. Future phases will add document-based retrieval (RAG) and autonomous agents with live financial data.

---
## Live Demo

🔗 **Live Demo:** https://finance-ai-assistant-ule5usq2zmervnpp8f7bvk.streamlit.app/
## What It Does

The assistant runs as a single chat interface that handles three types of requests within one ongoing conversation:

- **Finance Q&A** — ask any finance, accounting, or investment question and get a structured, expert-level answer
- **Concept Explanation** — request an explanation of any financial concept at beginner, intermediate, or advanced level
- **Text Summarization** — paste financial text (earnings reports, articles, filings) and get a structured executive summary

The assistant remembers the entire conversation — you can ask follow-up questions ("explain that for a beginner instead", "what are the risks?", "compare it with X") without repeating context.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| LLM Orchestration | LangChain (LCEL) |
| LLM Provider | Groq API — Llama 3.1 70B |
| UI | Streamlit |
| Config | python-dotenv |

---

## Architecture

```
finance-ai-assistant/
│
├── app.py                       # Streamlit UI — no business logic
│
├── src/
│   ├── config/
│   │   └── settings.py          # Centralised configuration (model, temperature, tokens)
│   │
│   ├── llm/
│   │   └── groq_client.py       # ChatGroq instance factory
│   │
│   ├── prompts/
│   │   └── finance_prompts.py   # System persona and prompt templates
│   │
│   ├── chains/
│   │   └── memory_chain.py      # LCEL chain with conversation memory
│   │
│   └── utils/
│       └── helpers.py           # Input validation, error formatting
│
├── tests/
│   └── test_chains.py           # Prompt and validation unit tests
│
├── assets/
│   └── trend.png                # App icon
│
├── .env.example
├── requirements.txt
└── README.md
```

The design follows separation of concerns: the UI layer (`app.py`) only collects input and displays output. All prompt logic lives in `prompts/`, all LLM configuration in `llm/`, and the chain that connects them in `chains/`. This means future phases (RAG, agents) can be added as new modules without modifying existing code.

---

## How Memory Works

LangChain 0.3's recommended pattern for conversational memory in LCEL is to manually pass message history through `MessagesPlaceholder`, rather than using the legacy `ConversationBufferMemory` class.

```
Every message sent to the LLM looks like:

[System Prompt] → [Q1] → [A1] → [Q2] → [A2] → ... → [New Question]
```

- Conversation history is stored in Streamlit's `session_state` as a list of `HumanMessage` / `AIMessage` objects
- On every new message, the full history is passed to the chain along with the new question
- The LLM itself is stateless — the application is responsible for maintaining and replaying context
- History is in-memory only (per browser session) and is cleared when the user clicks "New Conversation" or refreshes the page

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Heer117/finance-ai-assistant.git
cd finance-ai-assistant
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and add your Groq API key (get one free at [console.groq.com](https://console.groq.com)):

```
GROQ_API_KEY=your_actual_key_here
GROQ_MODEL_NAME=llama-3.1-70b-versatile
```

### 5. Run the app

```bash
streamlit run app.py
```

---

## Running Tests

```bash
pytest tests/ -v
```

Tests cover prompt template formatting and input validation logic — no API calls are made during testing.

---

## Project Status

| Feature | Status |
|---|---|
| Finance Q&A | Done |
| Concept Explanation | Done |
| Text Summarization | Done |
| Conversational memory (all tasks) | Done |
| Document upload + RAG | Planned |
| Live market data via agents/tools | Planned |

---

## Roadmap

**Phase 1 — Foundation** *(Completed)*
Stateless LLM chains for Q&A, explanation, and summarization with finance-specific prompt engineering.

**Phase 2 — Conversational Memory** *(Completed)*
Unified chat interface where all three capabilities share conversation history, enabling natural follow-up questions.

**Phase 3 — RAG** *(Planned)*
Upload financial documents (PDFs, reports) and ask questions answered from their actual content using a vector database.

**Phase 4 — Agents and Live Data** *(Planned)*
Tool-using agents that fetch live stock prices, perform financial calculations, and search recent news.

---

## Disclaimer

This tool is for educational and informational purposes only. It does not constitute financial advice. Always consult a qualified financial professional before making investment decisions.

---

## License

MIT
