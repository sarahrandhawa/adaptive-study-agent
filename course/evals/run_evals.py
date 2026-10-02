import json
from pathlib import Path

EVAL_DIR = Path(__file__).parent
TRACES_FILE = EVAL_DIR / "traces.jsonl"


def load_traces():
    traces = []

    with open(TRACES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                traces.append(json.loads(line))

    return traces


# CHECK 1:
# Catches the empty-response failure we observed in traces 8, 9, and 10.
def check_non_empty_answer(trace):
    answer = trace.get("assistant_output", "").strip()

    if not answer:
        return False, "Agent returned an empty answer."

    return True, "Answer is non-empty."


# CHECK 2:
# A successful agent response should contain evidence that search_docs ran.
def check_tool_execution(trace):
    steps = trace.get("steps", [])

    used_search_docs = any(
        step.get("tool") == "search_docs"
        for step in steps
    )

    if not used_search_docs:
        return False, "No search_docs execution was recorded."

    return True, "search_docs execution was recorded."


def run_evals():
    traces = load_traces()

    results = []

    for trace in traces:
        non_empty_pass, non_empty_reason = check_non_empty_answer(trace)
        tool_pass, tool_reason = check_tool_execution(trace)

        trace_pass = non_empty_pass and tool_pass

        results.append({
            "id": trace["id"],
            "non_empty_answer": non_empty_pass,
            "non_empty_reason": non_empty_reason,
            "tool_execution": tool_pass,
            "tool_reason": tool_reason,
            "overall_pass": trace_pass,
        })

    total = len(results)
    passed = sum(result["overall_pass"] for result in results)
    pass_rate = (passed / total * 100) if total else 0

    print("\nTRACE EVALUATION RESULTS")
    print("=" * 50)

    for result in results:
        status = "PASS" if result["overall_pass"] else "FAIL"
        print(f"Trace {result['id']:02}: {status}")

        if not result["non_empty_answer"]:
            print(f"  - {result['non_empty_reason']}")

        if not result["tool_execution"]:
            print(f"  - {result['tool_reason']}")

    print("=" * 50)
    print(f"Passed: {passed}/{total}")
    print(f"Pass rate: {pass_rate:.1f}%")

    return results


if __name__ == "__main__":
    run_evals()