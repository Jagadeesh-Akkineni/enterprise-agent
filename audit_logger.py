import json
import os
from datetime import datetime, timezone
from pathlib import Path

AUDIT_FILE = Path(__file__).parent / "audit_log.jsonl"


def log_query(query: str, result: dict) -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "reformulated_query": result.get("reformulated_query"),
        "can_answer": result.get("can_answer", False),
        "answer": result.get("answer"),
        "citations": result.get("citations", []),
        "retry_count": result.get("retry_count", 0),
        "chunks_retrieved": len(result.get("retrieved_chunks", [])),
    }
    with open(AUDIT_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
