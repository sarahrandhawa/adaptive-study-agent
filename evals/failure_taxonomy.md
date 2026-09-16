# TRACE Failure Taxonomy

This taxonomy was created after manually reviewing 20 capstone agent traces.

## 1. Empty agent response / no execution

The agent sometimes returns an empty answer and records no tool-use steps, leaving the user without an answer, refusal, or explanation.

Observed examples: Traces 8, 9, 10.

Impact: High

## 2. Noisy or weakly relevant retrieval

For several questions, the retrieved document IDs are potentially relevant, but the visible retrieved text begins with content that appears unrelated to the user's specific question. This makes grounding harder to verify and may indicate imprecise chunk retrieval.

Observed examples: Traces 1, 2, 3, 18, with some additional noisy retrieval visible in other traces.

Impact: Medium

## 3. Excessive response detail

Some responses provide substantially more information than the question requires. The answer may still be useful, but unnecessary detail makes simple FAQ responses less concise.

Observed examples: Traces 4 and 19.

Impact: Low

## 4. Potentially incomplete broad-topic answer

When asked a broad question, the agent may retrieve and present only a subset of the relevant information rather than making the scope or incompleteness clear.

Observed example: Trace 6.

Impact: Medium

# Initial Priority Ranking

1. Empty agent response / no execution — High impact
2. Noisy or weakly relevant retrieval — Medium impact
3. Potentially incomplete broad-topic answer — Medium impact
4. Excessive response detail — Low impact

# Top Target

Empty agent response / no execution.

This is the highest-priority failure because an empty response gives the user no useful result at all. It is also suitable for a deterministic code-based evaluation because an empty response can be detected reliably without subjective judgment.