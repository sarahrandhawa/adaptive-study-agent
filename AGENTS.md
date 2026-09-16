# Northwind FAQ Agent Rules

## Critical Rules

- Ground Northwind policy answers in retrieved company documents. Do not invent company policies or policy details.
- If the retrieved documents do not contain enough information to answer a question, say so clearly.
- Durable memory is user-specific state and is separate from Northwind company knowledge stored in the RAG system.
- Only persist approved stable user preferences: `preferred_name`, `preferred_answer_style`, and `preferred_language`.
- Do not store transient task instructions, raw tool output, credentials, secrets, or other unnecessary information in durable memory.
- Durable memory must be retrieved by `user_id` before an agent turn and only relevant preferences should influence the response.
- All memory writes must pass through the approved memory write gate.