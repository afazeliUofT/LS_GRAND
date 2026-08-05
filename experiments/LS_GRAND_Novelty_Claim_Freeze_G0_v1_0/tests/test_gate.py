from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from lsgrand_novelty.audit import EXPECTED_BASE_COMMIT, classify, load_inputs, package_root, run_gate, validate_inputs


class NoveltyGateTests(unittest.TestCase):
    def setUp(self):
        self.root = package_root()
        self.inputs = load_inputs(self.root)

    def test_inputs_validate(self):
        self.assertEqual(validate_inputs(self.inputs), [])

    def test_threshold_algorithm_boundary_is_frozen(self):
        c = {x["claim_id"]: x for x in self.inputs["claims"]["claims"]}
        self.assertEqual(c["C02"]["classification"], "ANTICIPATED_BY_THRESHOLD_ALGORITHM")
        self.assertTrue(c["C02"]["disposition"].startswith("BLOCK"))

    def test_narrow_receiver_is_only_provisional(self):
        c = {x["claim_id"]: x for x in self.inputs["claims"]["claims"]}
        self.assertEqual(c["C06"]["disposition"], "ALLOW_PROVISIONALLY")
        self.assertEqual(c["C07"]["disposition"], "ALLOW_PROVISIONALLY")

    def test_field_defining_is_not_authorized(self):
        v = classify(self.inputs)
        self.assertFalse(v["field_defining_program_authorized"])
        self.assertFalse(v["patent_freedom_to_operate_determined"])

    def test_expected_gate_verdict(self):
        v = classify(self.inputs)
        self.assertEqual(v["verdict"], "PASS_NARROWED_APPROXIMATE_CLAIM_FREEZE")
        self.assertTrue(v["next_bounded_numerical_gate_authorized"])

    def test_base_commit_is_frozen(self):
        self.assertEqual(self.inputs["scope"]["base_commit"], EXPECTED_BASE_COMMIT)

    def test_gate_renders_and_validates_core_files(self):
        with tempfile.TemporaryDirectory() as td:
            run = run_gate(Path(td), "unit_gate", "gate")
            self.assertTrue((run / "FINAL_G0_NOVELTY_VERDICT.json").is_file())
            v = json.loads((run / "FINAL_G0_NOVELTY_VERDICT.json").read_text())
            self.assertEqual(v["verdict"], "PASS_NARROWED_APPROXIMATE_CLAIM_FREEZE")

    def test_blocked_language_is_explicit(self):
        blocked = self.inputs["language"]["blocked_phrases"]
        self.assertIn("new generic exact stopping theorem", blocked)
        self.assertIn("field-defining exact LS-GRAND", blocked)


if __name__ == "__main__":
    unittest.main()
