from __future__ import annotations

import itertools
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from lsgrand_decisive.affine import (
    affine_map_for_path,
    all_one_slip_paths,
    apply_affine_bits,
    linear_code_affine_collision,
)
from lsgrand_decisive.channel import (
    affine_transform_bits,
    fixed_slip_path,
    one_slip_hypotheses,
    simulate_qpsk_awgn,
)
from lsgrand_decisive.decoders import (
    batch_marginal_scores,
    empirical_true_ranks,
    latent_list_decode,
    latent_osd_batch,
)
from lsgrand_decisive.experiments import adjudicate, load_config
from lsgrand_decisive.gf2 import systematic_random_code
from lsgrand_decisive.oracle import exhaustive_oracle
from lsgrand_decisive.search import certified_lsgrand


class TestAffineOrbit(unittest.TestCase):
    def test_affine_matches_waveform_map(self) -> None:
        rng = np.random.default_rng(1)
        for n in (8, 12, 20):
            for _ in range(20):
                bits = rng.integers(0, 2, size=n, dtype=np.uint8)
                path = rng.integers(0, 4, size=n // 2, dtype=np.int8)
                self.assertTrue(np.array_equal(apply_affine_bits(bits, path), affine_transform_bits(bits, path)))
                a, b = affine_map_for_path(path)
                self.assertEqual(a.shape, (n, n))
                self.assertEqual(b.shape, (n,))

    def test_algebraic_collision_matches_enumeration(self) -> None:
        rng = np.random.default_rng(2)
        for family in ("dense", "sparse"):
            code = systematic_random_code(12, 8, rng, family)
            cws = code.enumerate_codewords(max_k=12)
            code_set = {np.packbits(c, bitorder="big").tobytes() for c in cws}
            for _, path in all_one_slip_paths(6):
                exact = linear_code_affine_collision(code, path).collision_fraction
                hits = sum(
                    np.packbits(apply_affine_bits(c, path), bitorder="big").tobytes() in code_set
                    for c in cws
                )
                self.assertAlmostEqual(exact, hits / len(cws), places=12)


class TestDecoders(unittest.TestCase):
    def test_batch_score_matches_direct_oracle(self) -> None:
        rng = np.random.default_rng(3)
        code = systematic_random_code(12, 8, rng, "dense")
        states = one_slip_hypotheses(6, 0.5, [1, 3])
        word = code.encode(rng.integers(0, 2, size=code.k, dtype=np.uint8))
        sample = simulate_qpsk_awgn(word, states[4].path, 4.0, rng)
        cands = code.enumerate_codewords(max_k=12)
        scores, evals, bitops, _ = batch_marginal_scores(sample.y, sample.n0, cands, states, batch_size=31)
        oracle = exhaustive_oracle(sample.y, sample.n0, code, states, max_k=12)
        winner = int(np.argmax(scores))
        self.assertEqual(int("".join(str(int(x)) for x in cands[winner]), 2), oracle.marginal_winner_int)
        self.assertEqual(evals, len(cands) * len(states))
        self.assertEqual(bitops, evals * code.n)

    def test_list_and_osd_return_codewords(self) -> None:
        rng = np.random.default_rng(4)
        code = systematic_random_code(24, 16, rng, "sparse")
        states = one_slip_hypotheses(12, 0.5, [1, 3])
        word = code.encode(rng.integers(0, 2, size=code.k, dtype=np.uint8))
        sample = simulate_qpsk_awgn(word, states[5].path, 7.0, rng)
        ls = latent_list_decode(sample.y, sample.n0, code, states, list_size=4, query_cap=10000)
        osd = latent_osd_batch(sample.y, sample.n0, code, states, order=1, pool_size=6, state_limit=None)
        self.assertTrue(ls.success)
        self.assertTrue(osd.success)
        self.assertTrue(code.contains(np.asarray(ls.decoded_bits, dtype=np.uint8)))
        self.assertTrue(code.contains(np.asarray(osd.decoded_bits, dtype=np.uint8)))

    def test_rank_probe_noiseless_like(self) -> None:
        rng = np.random.default_rng(5)
        code = systematic_random_code(16, 10, rng, "dense")
        states = one_slip_hypotheses(8, 0.5, [1, 3])
        path = fixed_slip_path(8, 2, 1)
        word = code.encode(rng.integers(0, 2, size=code.k, dtype=np.uint8))
        sample = simulate_qpsk_awgn(word, path, 20.0, rng)
        rec = empirical_true_ranks(sample.y, sample.n0, word, states, path, latent_cap=1000, plain_cap=1000)
        self.assertIsNotNone(rec.latent_true_component_rank)
        self.assertFalse(rec.latent_cap_hit)
        self.assertGreaterEqual(rec.rank_ratio_lower_bound or 0.0, 1.0)

    def test_certified_decoder_matches_oracle(self) -> None:
        rng = np.random.default_rng(6)
        for _ in range(5):
            code = systematic_random_code(10, 7, rng, "dense")
            states = one_slip_hypotheses(5, 0.5, [1, 3])
            word = code.encode(rng.integers(0, 2, size=code.k, dtype=np.uint8))
            true = states[int(rng.integers(len(states)))]
            sample = simulate_qpsk_awgn(word, true.path, 2.0, rng)
            oracle = exhaustive_oracle(sample.y, sample.n0, code, states, max_k=12)
            result = certified_lsgrand(sample.y, sample.n0, code, states, query_cap=len(states) * (1 << code.n))
            self.assertTrue(result.certified)
            if not oracle.marginal_tie:
                self.assertEqual(result.decoded_int, oracle.marginal_winner_int)


class TestPolicy(unittest.TestCase):
    def test_config_loads(self) -> None:
        root = Path(__file__).resolve().parents[1]
        cfg = load_config(root / "configs" / "smoke.json")
        self.assertEqual(cfg["profile"], "smoke")

    def test_failed_certificate_cannot_upgrade(self) -> None:
        root = Path(__file__).resolve().parents[1]
        cfg = load_config(root / "configs" / "gate.json")
        exact_summary = {
            "non_tied_trials": 1200, "cert_mismatches": 0, "certificate_failures_or_caps": 0,
            "direct_failures": 0, "trials": 1200, "marginal_first_switches": 120,
            "marginal_errors": 10, "first_errors": 8,
            "marginal_vs_first_mcnemar": {"one_sided_p_a_better": 1.0},
            "median_effective_path_multiplicity": 1.1,
        }
        rank = pd.DataFrame([
            {"location_label":"early","n":64,"family":"dense","latent_success_rate":1.0,"median_rank_ratio_lower_bound":100.0,"p10_rank_ratio_lower_bound":20.0},
            {"location_label":"middle","n":64,"family":"sparse","latent_success_rate":1.0,"median_rank_ratio_lower_bound":100.0,"p10_rank_ratio_lower_bound":20.0},
            {"location_label":"early","n":128,"family":"dense","latent_success_rate":1.0,"median_rank_ratio_lower_bound":100.0,"p10_rank_ratio_lower_bound":20.0},
            {"location_label":"middle","n":128,"family":"sparse","latent_success_rate":1.0,"median_rank_ratio_lower_bound":100.0,"p10_rank_ratio_lower_bound":20.0},
        ])
        # No performance comparison => F3 fails; certificate table also fails.
        verdict, _ = adjudicate(cfg, exact_summary, pd.DataFrame(), rank, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), True)
        self.assertNotIn("CONTINUE_SIGNIFICANT_BUT_NARROW", verdict["verdict"])
        self.assertEqual(verdict["verdict"], "PIVOT_THEORY_ONLY_OR_STOP_ALGORITHM")


if __name__ == "__main__":
    unittest.main()
