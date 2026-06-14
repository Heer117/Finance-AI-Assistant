"""
finance_prompts.py — Prompt Templates for 3 Core Tasks

Tasks:
  1. Finance Q&A
  2. Concept Explanation
  3. Text Summarization
"""

from langchain_core.prompts import ChatPromptTemplate

# Shared finance expert persona used across all tasks
BASE_FINANCE_PERSONA = """You are FinanceAI, a Senior Financial Analyst and Educator \
with 20 years of experience across investment banking, corporate finance, equity research, \
and financial planning.

Your expertise includes:
- Financial statement analysis (Income Statement, Balance Sheet, Cash Flow)
- Valuation methods (DCF, comparables, precedent transactions)
- Capital markets, fixed income, derivatives, and portfolio management
- Macroeconomics, monetary policy, and market dynamics
- Accounting standards (GAAP, IFRS) and financial ratios

Your communication style:
- Clear, precise, and educational
- Use concrete examples and analogies when explaining concepts
- Structure responses with headings and bullet points for readability
- Define technical terms when first used

Your constraints:
- ONLY respond to questions related to finance, economics, accounting, and investments
- If asked about non-financial topics, politely decline and redirect
- NEVER recommend specific stocks, funds, or investment products
- Add a brief disclaimer when answers involve interpretation or judgment"""


# ── 1. Finance Q&A ────────────────────────────────────────────────────────────

def get_qa_prompt() -> ChatPromptTemplate:
    """Prompt for answering specific finance questions."""
    return ChatPromptTemplate.from_messages([
        ("system", BASE_FINANCE_PERSONA),
        ("human", """Please answer the following finance question thoroughly and accurately.

Question: {question}

Provide your answer in this structure:

**Direct Answer:** (1-2 sentences with the core answer)

**Detailed Explanation:** (3-5 sentences expanding on the concept)

**Key Points:**
- (Bullet point 1)
- (Bullet point 2)
- (Bullet point 3 if applicable)

**Example:** (A practical, real-world example if helpful)

**Important Note:** (Any caveats or disclaimers)"""),
    ])


# ── 2. Concept Explanation ────────────────────────────────────────────────────

def get_explanation_prompt() -> ChatPromptTemplate:
    """Prompt for explaining financial concepts at beginner/intermediate/advanced level."""
    return ChatPromptTemplate.from_messages([
        ("system", BASE_FINANCE_PERSONA),
        ("human", """Explain the financial concept "{concept}" for a {level} audience.

## What is {concept}?
(Clear definition in 2-3 sentences)

## The Simple Version
(Explain using an everyday analogy — no jargon)

## How It Works
(Step-by-step explanation)

## Why It Matters
(Real-world significance and use cases)

## Key Formula or Components
(If applicable)

## Related Concepts
(2-3 connected concepts with one-line descriptions)"""),
    ])


# ── 3. Text Summarization ─────────────────────────────────────────────────────

def get_summarization_prompt() -> ChatPromptTemplate:
    """Prompt for summarizing financial documents or articles."""
    return ChatPromptTemplate.from_messages([
        ("system", BASE_FINANCE_PERSONA),
        ("human", """Summarize the following financial text into a clear, structured format.

**Financial Text:**
{text}

**Provide a summary in this structure:**

## Executive Summary
(3-4 sentences capturing the most important information)

## Key Financial Metrics Mentioned
(List every specific number, percentage, or financial figure)
- Metric: Value (context)

## Main Topics Covered
(Bullet list of primary subjects discussed)

## Critical Findings or Announcements
(Significant conclusions or news)

## Risks and Concerns Highlighted
(Any negative factors or warnings mentioned)"""),
    ])