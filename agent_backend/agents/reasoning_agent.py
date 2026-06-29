import os

from dotenv import load_dotenv
from langchain_cohere import ChatCohere
from langchain_core.messages import HumanMessage

from agents.state import AgentState
from config import REASONING_MAX_TOKENS, REASONING_MODEL

load_dotenv()

_llm = None

REASONING_PROMPT = """You are an enterprise knowledge assistant. Answer the employee's question \
using ONLY the document excerpts provided below.

Rules:
- Base your answer strictly on the excerpts. Do not add outside knowledge.
- If the excerpts do not contain enough information, respond with exactly: CANNOT_ANSWER
- Be concise and professional.

Question: {query}

Document Excerpts:
{context}

Answer:"""


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatCohere(
            model=REASONING_MODEL,
            cohere_api_key=os.getenv("COHERE_API_KEY"),
            max_tokens=REASONING_MAX_TOKENS,
        )
    return _llm


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return " ".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ).strip()
    return str(content).strip()


def reasoning_node(state: AgentState) -> AgentState:
    llm = _get_llm()
    chunks = state.get("retrieved_chunks", [])
    query = state["query"]

    context = "\n\n---\n\n".join(
        f"[{c['metadata'].get('source_file', 'Unknown')} — {c['metadata'].get('section_title', '')}]\n{c['text']}"
        for c in chunks
    )

    prompt = REASONING_PROMPT.format(query=query, context=context)
    response = llm.invoke([HumanMessage(content=prompt)])
    answer = _extract_text(response.content)

    can_answer = answer != "CANNOT_ANSWER"

    return {
        **state,
        "answer": answer if can_answer else "I cannot find an answer to this question in the available enterprise documents.",
        "can_answer": can_answer,
    }


def check_can_answer(state: AgentState) -> str:
    return "citation" if state.get("can_answer", False) else "end"
