from typing import TypedDict, List, Optional


class AgentState(TypedDict):
    query: str
    reformulated_query: Optional[str]
    retrieved_chunks: List[dict]
    retrieval_scores: List[float]
    retry_count: int
    answer: Optional[str]
    citations: List[dict]
    can_answer: bool
