import json
from pathlib import Path
from typing import Any


MEMORY_FILE = Path(__file__).parent / "memory.json"

ALLOWED_MEMORY_KEYS = {
    "preferred_name",
    "preferred_answer_style",
    "preferred_language",
    "topic_mastery",
}

VALID_VERDICTS = {"correct", "partially_correct", "incorrect"}


def _load_memory() -> dict:
    """Load all saved memory from disk."""
    if not MEMORY_FILE.exists():
        return {}

    try:
        with MEMORY_FILE.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError, TypeError):
        return {}

    return data if isinstance(data, dict) else {}


def _save_memory(memory: dict) -> None:
    """Write all memory to disk."""
    with MEMORY_FILE.open("w", encoding="utf-8") as file:
        json.dump(memory, file, indent=2)


def get_memory(user_id: str) -> dict:
    """Return the saved memory for one user."""
    memory = _load_memory()
    user_memory = memory.get(user_id, {})
    return user_memory if isinstance(user_memory, dict) else {}


def save_memory(user_id: str, key: str, value: Any) -> None:
    """Save or replace one approved durable memory for a user."""

    if key not in ALLOWED_MEMORY_KEYS:
        raise ValueError(
            f"'{key}' is not an approved durable memory."
        )

    if key == "topic_mastery" and not isinstance(value, dict):
        raise ValueError("topic_mastery must be an object keyed by topic.")

    memory = _load_memory()

    if user_id not in memory or not isinstance(memory[user_id], dict):
        memory[user_id] = {}

    memory[user_id][key] = value

    _save_memory(memory)


def list_memories() -> dict:
    """Return all currently stored memories."""
    return _load_memory()


def _coerce_count(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def empty_topic_entry() -> dict[str, Any]:
    return {
        "attempts": 0,
        "correct": 0,
        "partial": 0,
        "incorrect": 0,
        "weak_concepts": [],
    }


def normalize_topic_entry(raw: Any) -> dict[str, Any]:
    """Coerce messy/corrupt topic stats into the expected shape."""

    entry = empty_topic_entry()
    if not isinstance(raw, dict):
        return entry

    entry["attempts"] = _coerce_count(raw.get("attempts"))
    entry["correct"] = _coerce_count(raw.get("correct"))
    entry["partial"] = _coerce_count(raw.get("partial"))
    entry["incorrect"] = _coerce_count(raw.get("incorrect"))

    weak_raw = raw.get("weak_concepts")
    cleaned: list[str] = []
    seen: set[str] = set()
    if isinstance(weak_raw, list):
        for item in weak_raw:
            if not isinstance(item, str):
                continue
            concept = item.strip()
            key = concept.casefold()
            if concept and key not in seen:
                seen.add(key)
                cleaned.append(concept)
    entry["weak_concepts"] = cleaned
    return entry


def mastery_ratio(entry: dict[str, Any]) -> float | None:
    """(correct + 0.5 * partial) / attempts, or None when unassessed."""

    normalized = normalize_topic_entry(entry)
    attempts = normalized["attempts"]
    if attempts == 0:
        return None
    return (normalized["correct"] + 0.5 * normalized["partial"]) / attempts


def mastery_label(entry: dict[str, Any]) -> str:
    """Deterministic label from recorded counts — not an LLM score."""

    ratio = mastery_ratio(entry)
    if ratio is None:
        return "Not assessed"
    if ratio >= 0.8:
        return "Strong"
    if ratio >= 0.6:
        return "Developing"
    return "Needs review"


def topic_progress(topic: str, entry: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_topic_entry(entry)
    return {
        "topic": topic,
        "mastery": mastery_label(normalized),
        "attempts": normalized["attempts"],
        "correct": normalized["correct"],
        "partial": normalized["partial"],
        "incorrect": normalized["incorrect"],
        "weak_concepts": list(normalized["weak_concepts"]),
    }


def _topic_mastery_map(user_id: str) -> dict[str, Any]:
    raw = get_memory(user_id).get("topic_mastery")
    if not isinstance(raw, dict):
        return {}
    return raw


def get_topic_entry(user_id: str, topic: str) -> dict[str, Any]:
    normalized_topic = topic.strip()
    if not normalized_topic:
        return empty_topic_entry()
    return normalize_topic_entry(_topic_mastery_map(user_id).get(normalized_topic))


def weak_concepts_for_topic(user_id: str, topic: str) -> list[str]:
    return list(get_topic_entry(user_id, topic)["weak_concepts"])


def record_quiz_result(
    *,
    user_id: str,
    topic: str,
    verdict: str,
    identified_gap: str | None = None,
) -> dict[str, Any]:
    """Update compact topic_mastery through the write gate. Returns UI-ready progress."""

    normalized_topic = topic.strip()
    if not normalized_topic:
        raise ValueError("topic must not be empty")
    if verdict not in VALID_VERDICTS:
        raise ValueError("verdict must be correct, partially_correct, or incorrect")

    mastery = dict(_topic_mastery_map(user_id))
    entry = normalize_topic_entry(mastery.get(normalized_topic))
    entry["attempts"] += 1
    if verdict == "correct":
        entry["correct"] += 1
    elif verdict == "partially_correct":
        entry["partial"] += 1
    else:
        entry["incorrect"] += 1

    gap = identified_gap.strip() if isinstance(identified_gap, str) else ""
    if gap:
        existing = {item.casefold() for item in entry["weak_concepts"]}
        if gap.casefold() not in existing:
            entry["weak_concepts"].append(gap)

    mastery[normalized_topic] = entry
    save_memory(user_id, "topic_mastery", mastery)
    return topic_progress(normalized_topic, entry)
