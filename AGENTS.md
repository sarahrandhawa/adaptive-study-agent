# Northwind FAQ Agent Rules

## Critical Rules

- Ground Northwind policy answers in retrieved company documents. Do not invent company policies or policy details.
- If the retrieved documents do not contain enough information to answer a question, say so clearly.
- Durable memory is user-specific state and is separate from Northwind company knowledge stored in the RAG system.
- Only persist approved durable memory: `preferred_name`, `preferred_answer_style`, `preferred_language`, and compact `topic_mastery` learning stats.
- `topic_mastery` is derived from recorded quiz verdicts, not LLM-invented percentages.
- Do not store quiz questions, student answers, retrieved chunks, credentials, secrets, or conversation history in durable memory.
- Durable memory must be retrieved by `user_id` before an agent turn and only relevant preferences should influence the agent response.
- All memory writes must pass through the approved memory write gate.