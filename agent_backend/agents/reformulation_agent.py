import os

from dotenv import load_dotenv
from langchain_cohere import ChatCohere
from langchain_core.messages import HumanMessage

from agents.state import AgentState
from config import REFORMULATION_MAX_TOKENS, REFORMULATION_MODEL

load_dotenv()

_llm = None

REFORMULATE_PROMPT = """Rephrase the following employee question to make it clearer and more specific \
for searching an enterprise HR document database. Return only the rephrased question, nothing else.

Original question: {query}

Rephrased question:"""


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatCohere(
            model=REFORMULATION_MODEL,
            cohere_api_key=os.getenv("COHERE_API_KEY"),
            max_tokens=REFORMULATION_MAX_TOKENS,
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


def reformulation_node(state: AgentState) -> AgentState:
    llm = _get_llm()
    prompt = REFORMULATE_PROMPT.format(query=state["query"])
    response = llm.invoke([HumanMessage(content=prompt)])
    reformulated = _extract_text(response.content)

    return {
        **state,
        "reformulated_query": reformulated,
        "retry_count": state.get("retry_count", 0) + 1,
    }
