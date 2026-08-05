from __future__ import annotations

import unittest

import numpy as np

from lsgrand.search import logaddexp, logsumexp


class TestCertificateAlgebra(unittest.TestCase):
    def test_logsumexp(self) -> None:
        vals = [-1000.0, -1001.0, -1002.0]
        m = max(vals)
        expected = m + np.log(sum(np.exp(v - m) for v in vals))
        self.assertAlmostEqual(logsumexp(vals), expected, places=12)
        self.assertAlmostEqual(logaddexp(-3.0, -4.0), np.log(np.exp(-3.0) + np.exp(-4.0)), places=12)

    def test_adversarial_first_witness_logic(self) -> None:
        # Codeword A has the largest single witness, but B has larger marginal
        # mass after two state contributions.  This is the minimal reason a
        # first-hit or max-path decision is not generally marginal ML.
        a = [0.46, 0.01]
        b = [0.30, 0.29]
        self.assertGreater(max(a), max(b))
        self.assertLess(sum(a), sum(b))

    def test_head_bound_contains_unseen_mass(self) -> None:
        # Each label occurs at most once in each state queue.  Any unrevealed
        # contribution is no larger than that queue's head.
        queues = [
            [("a", 0.5), ("b", 0.3), ("c", 0.1)],
            [("b", 0.4), ("c", 0.2), ("a", 0.05)],
        ]
        revealed = [1, 1]
        heads = [q[k][1] for q, k in zip(queues, revealed)]
        partial = {"a": 0.5, "b": 0.4}
        seen = {"a": {0}, "b": {1}}
        for label in ("a", "b", "c"):
            tail = sum(heads[s] for s in range(2) if s not in seen.get(label, set()))
            exact_unseen = sum(
                weight
                for s, q in enumerate(queues)
                for lab, weight in q[revealed[s] :]
                if lab == label
            )
            self.assertLessEqual(exact_unseen, tail + 1e-12)


if __name__ == "__main__":
    unittest.main()
