from audit_logger import log_query
from graph.workflow import app


INITIAL_STATE = {
    "query": "",
    "reformulated_query": None,
    "retrieved_chunks": [],
    "retrieval_scores": [],
    "retry_count": 0,
    "answer": None,
    "citations": [],
    "can_answer": False,
}


def run_query(query: str) -> dict:
    state = {**INITIAL_STATE, "query": query}
    result = app.invoke(state)
    log_query(query, result)
    return result


def format_response(result: dict) -> str:
    answer = result.get("answer", "No answer generated.")
    citations = result.get("citations", [])
    reformulated = result.get("reformulated_query")

    lines = []

    if reformulated:
        lines.append(f"(Query interpreted as: \"{reformulated}\")\n")

    lines.append(f"Answer:\n{answer}")

    if citations:
        lines.append("\nSources:")
        for i, c in enumerate(citations, 1):
            page = f"Page {c['page_start']}" if c.get("page_start") else ""
            if c.get("page_end") and c["page_end"] != c["page_start"]:
                page += f"–{c['page_end']}"
            page_str = f" ({page})" if page else ""
            lines.append(f"  {i}. {c['source_file']} — {c['section_title']}{page_str}")

    return "\n".join(lines)


if __name__ == "__main__":
    print("Enterprise Knowledge Assistant")
    print("=" * 40)
    print("Type your question and press Enter. Type 'quit' to exit.\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        print("\nProcessing...\n")
        result = run_query(query)
        print(format_response(result))
        print()
