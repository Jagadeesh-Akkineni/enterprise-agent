from langgraph.graph import END, START, StateGraph

from agents.citation_agent import citation_node
from agents.reasoning_agent import check_can_answer, reasoning_node
from agents.reformulation_agent import reformulation_node
from agents.retrieval_agent import check_retrieval_quality, retrieval_node
from agents.state import AgentState


def build_graph():
    graph = StateGraph(AgentState)

    # --- Nodes ---
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("reformulate", reformulation_node)
    graph.add_node("reasoning", reasoning_node)
    graph.add_node("citation", citation_node)

    # --- Edges ---
    graph.add_edge(START, "retrieval")

    # After retrieval: go to reasoning if results are good, else reformulate
    graph.add_conditional_edges(
        "retrieval",
        check_retrieval_quality,
        {"reformulate": "reformulate", "reasoning": "reasoning"},
    )

    # After reformulation: retry retrieval with the new query
    graph.add_edge("reformulate", "retrieval")

    # After reasoning: attach citations if answered, else end
    graph.add_conditional_edges(
        "reasoning",
        check_can_answer,
        {"citation": "citation", "end": END},
    )

    graph.add_edge("citation", END)

    return graph.compile()


app = build_graph()
