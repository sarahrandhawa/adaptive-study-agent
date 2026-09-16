import json
import requests
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000"
AGENT_URL = f"{BASE_URL}/agent"

EVAL_DIR = Path(__file__).parent
QUESTIONS_FILE = EVAL_DIR / "trace_questions.json"
OUTPUT_FILE = EVAL_DIR / "traces.jsonl"


def load_questions():
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_traces():
    questions = load_questions()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as output:
        for item in questions:
            trace_id = item["id"]
            question = item["question"]

            print(f"[{trace_id}/20] {question}")

            try:
                response = requests.post(
                    AGENT_URL,
                    json={"message": question},
                    timeout=60,
                )
                response.raise_for_status()
                result = response.json()

                trace = {
                    "id": trace_id,
                    "user_input": question,
                    "assistant_output": result.get("answer", ""),
                    "model": result.get("model"),
                    "steps": result.get("steps", []),
                }

            except Exception as exc:
                trace = {
                    "id": trace_id,
                    "user_input": question,
                    "assistant_output": "",
                    "model": None,
                    "steps": [],
                    "collection_error": str(exc),
                }

            output.write(json.dumps(trace) + "\n")

    print(f"\nDone. Saved {len(questions)} traces to {OUTPUT_FILE}")


if __name__ == "__main__":
    collect_traces()