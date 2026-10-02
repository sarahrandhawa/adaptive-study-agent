# Golden set (M5, DEC-019 / DEC-025)

Test data for evals/run_golden.py. Not real user notes; ingest only under the `eval` identity.

- water_cycle_notes.txt: synthetic study notes with 3 invented course terms (Harlow ratio, Kestrel line, three-millimeter rule) and one injected instruction in section 4.
- quiz_topics.json: 10 question-writing cases for EVAL-1 (each section topic twice to exercise "no repeats", one focus case, one off-document topic).
- grade_items.jsonl: 20 answers for EVAL-2. Fill in "label" with correct / partially_correct / incorrect BEFORE opening the intent file.
- ask_items.jsonl: 8 cases for EVAL-3.
- _intent_DO_NOT_OPEN_UNTIL_LABELED.json: the drafter's intended answer types; compare only after labeling.
