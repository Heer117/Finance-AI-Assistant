"""
finance_chains.py — LangChain Chain Definitions (3 Tasks)

Each chain follows the same LCEL pattern:
  Prompt Template | LLM | Output Parser

Chains defined:
  1. QA Chain
  2. Explanation Chain
  3. Summarization Chain
"""

from langchain_core.output_parsers import StrOutputParser

from src.llm.groq_client import (
    get_qa_llm,
    get_explain_llm,
    get_summarize_llm,
)
from src.prompts.finance_prompts import (
    get_qa_prompt,
    get_explanation_prompt,
    get_summarization_prompt,
)

# StrOutputParser converts AIMessage → plain string
_output_parser = StrOutputParser()


def build_qa_chain():
    """
    Chain for Finance Q&A.
    Input:  {"question": str}
    Output: str
    """
    return get_qa_prompt() | get_qa_llm() | _output_parser


def build_explanation_chain():
    """
    Chain for Concept Explanation.
    Input:  {"concept": str, "level": str}
    Output: str
    """
    return get_explanation_prompt() | get_explain_llm() | _output_parser


def build_summarization_chain():
    """
    Chain for Text Summarization.
    Input:  {"text": str}
    Output: str
    """
    return get_summarization_prompt() | get_summarize_llm() | _output_parser