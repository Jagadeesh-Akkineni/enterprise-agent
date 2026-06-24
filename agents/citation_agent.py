from agents.state import AgentState


def citation_node(state: AgentState) -> AgentState:
    chunks = state.get("retrieved_chunks", [])
    seen = set()
    citations = []

    for chunk in chunks:
        meta = chunk.get("metadata", {})
        key = (meta.get("source_file"), meta.get("section_title"))
        if key in seen:
            continue
        seen.add(key)
        citations.append({
            "source_file": meta.get("source_file", "Unknown"),
            "section_title": meta.get("section_title", ""),
            "page_start": meta.get("page_start"),
            "page_end": meta.get("page_end"),
        })

    return {**state, "citations": citations}
