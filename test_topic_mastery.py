"""Deterministic tests for topic mastery labels and memory updates."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import memory_store as ms


class TopicMasteryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._original_file = ms.MEMORY_FILE
        handle = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        handle.write(b"{}")
        handle.close()
        self._temp_path = Path(handle.name)
        ms.MEMORY_FILE = self._temp_path

    def tearDown(self) -> None:
        ms.MEMORY_FILE = self._original_file
        self._temp_path.unlink(missing_ok=True)

    def test_mastery_labels(self) -> None:
        self.assertEqual(ms.mastery_label({"attempts": 0}), "Not assessed")
        self.assertEqual(
            ms.mastery_label({"attempts": 5, "correct": 4, "partial": 0, "incorrect": 1}),
            "Strong",
        )
        self.assertEqual(
            ms.mastery_label({"attempts": 5, "correct": 2, "partial": 2, "incorrect": 1}),
            "Developing",
        )
        self.assertEqual(
            ms.mastery_label({"attempts": 1, "correct": 0, "partial": 0, "incorrect": 1}),
            "Needs review",
        )

    def test_mastery_ratio(self) -> None:
        self.assertIsNone(ms.mastery_ratio({"attempts": 0}))
        self.assertEqual(ms.mastery_ratio({"attempts": 2, "correct": 1, "partial": 0}), 0.5)
        self.assertEqual(ms.mastery_ratio({"attempts": 2, "correct": 1, "partial": 1}), 0.75)

    def test_record_quiz_result_counts_and_gap(self) -> None:
        first = ms.record_quiz_result(
            user_id="demo-user",
            topic="Japanese greetings",
            verdict="incorrect",
            identified_gap="Japanese time-of-day greetings",
        )
        self.assertEqual(first["attempts"], 1)
        self.assertEqual(first["incorrect"], 1)
        self.assertEqual(first["mastery"], "Needs review")
        self.assertEqual(first["weak_concepts"], ["Japanese time-of-day greetings"])

        second = ms.record_quiz_result(
            user_id="demo-user",
            topic="Japanese greetings",
            verdict="incorrect",
            identified_gap="Japanese time-of-day greetings",
        )
        self.assertEqual(second["attempts"], 2)
        self.assertEqual(second["incorrect"], 2)
        self.assertEqual(second["weak_concepts"], ["Japanese time-of-day greetings"])

        third = ms.record_quiz_result(
            user_id="demo-user",
            topic="Japanese greetings",
            verdict="correct",
            identified_gap=None,
        )
        self.assertEqual(third["correct"], 1)
        self.assertEqual(third["attempts"], 3)
        self.assertEqual(third["weak_concepts"], ["Japanese time-of-day greetings"])

    def test_preferences_survive_mastery_writes(self) -> None:
        ms.save_memory("demo-user", "preferred_language", "Japanese")
        ms.record_quiz_result(
            user_id="demo-user",
            topic="Japanese greetings",
            verdict="partially_correct",
            identified_gap="Particle は vs が",
        )
        stored = ms.get_memory("demo-user")
        self.assertEqual(stored["preferred_language"], "Japanese")
        self.assertIn("topic_mastery", stored)
        self.assertNotIn("preferred_name", stored)

    def test_corrupt_topic_mastery_is_normalized(self) -> None:
        ms.save_memory("demo-user", "preferred_name", "Sarah")
        payload = {
            "demo-user": {
                "preferred_name": "Sarah",
                "topic_mastery": {
                    "Japanese greetings": "not-an-object",
                },
            }
        }
        self._temp_path.write_text(json.dumps(payload), encoding="utf-8")
        entry = ms.get_topic_entry("demo-user", "Japanese greetings")
        self.assertEqual(entry["attempts"], 0)
        self.assertEqual(ms.weak_concepts_for_topic("demo-user", "Japanese greetings"), [])

        updated = ms.record_quiz_result(
            user_id="demo-user",
            topic="Japanese greetings",
            verdict="correct",
        )
        self.assertEqual(updated["attempts"], 1)
        self.assertEqual(updated["correct"], 1)
        self.assertEqual(updated["weak_concepts"], [])
        self.assertEqual(ms.get_memory("demo-user")["preferred_name"], "Sarah")

    def test_new_user_has_no_weak_concepts(self) -> None:
        self.assertEqual(ms.get_memory("brand-new-user"), {})
        self.assertEqual(ms.weak_concepts_for_topic("brand-new-user", "Japanese greetings"), [])

    def test_unknown_memory_key_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ms.save_memory("demo-user", "raw_answer", "konnichiwa")


if __name__ == "__main__":
    unittest.main()
