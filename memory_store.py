import json
from pathlib import Path
from typing import Any


MEMORY_FILE = Path(__file__).parent / "memory.json"

ALLOWED_MEMORY_KEYS = {
    "preferred_name",
    "preferred_answer_style",
    "preferred_language",
}

def _load_memory() -> dict:
    """Load all saved memory from disk."""
    if not MEMORY_FILE.exists():
        return {}

    with MEMORY_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def _save_memory(memory: dict) -> None:
    """Write all memory to disk."""
    with MEMORY_FILE.open("w", encoding="utf-8") as file:
        json.dump(memory, file, indent=2)


def get_memory(user_id: str) -> dict:
    """Return the saved memory for one user."""
    memory = _load_memory()
    return memory.get(user_id, {})


def save_memory(user_id: str, key: str, value: Any) -> None:
    """Save or replace one approved durable memory for a user."""

    if key not in ALLOWED_MEMORY_KEYS:
        raise ValueError(
            f"'{key}' is not an approved durable memory."
        )

    memory = _load_memory()

    if user_id not in memory:
        memory[user_id] = {}

    memory[user_id][key] = value

    _save_memory(memory)


def list_memories() -> dict:
    """Return all currently stored memories."""
    return _load_memory()