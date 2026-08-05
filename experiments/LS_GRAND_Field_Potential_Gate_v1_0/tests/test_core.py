from __future__ import annotations

import itertools
import unittest

import numpy as np

from lsgrand.channel import (
    affine_transform_bits,
    bits_to_qpsk,
    fixed_slip_path,
    one_slip_hypotheses,
    qpsk_hard_bits,
    qpsk_loglikelihood,
    simulate_qpsk_awgn,
    stream_parameters,
)
from lsgrand.diagnostics import canonical_collision_paths, exact_orbit_collision_records, rank_separation_record
from lsgrand.gf2 import inverse, most_reliable_basis_generator, rank, systematic_random_code
from lsgrand.oracle import direct_probability_scores, exhaustive_oracle
from lsgrand.search import SubsetSumEnumerator, certified_lsgrand, latent_osd


class TestGF2(unittest.TestCase):
    def test_inverse_and_code(self) -> None:
        a = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=np.uint8)
        # The displayed matrix has rank 2; verify singular handling first.
        self.assertEqual(rank(a), 2)
        with self.assertRaises(ValueError):
            inverse(a)
        b = np.array([[1, 1, 0], [0, 1, 1], [1, 1, 1]], dtype=np.uint8)
        bi = inverse(b)
        self.assertTrue(np.array_equal((bi @ b) & 1, np.eye(3, dtype=np.uint8)))
        rng = np.random.default_rng(1)
        for family in ("dense", "sparse"):
            code = systematic_random_code(16, 12, rng, family)
            for _ in range(20):
                u = rng.integers(0, 2, size=code.k, dtype=np.uint8)
                c = code.encode(u)
                self.assertTrue(code.contains(c))
                bad = c.copy()
                bad[int(rng.integers(code.n))] ^= 1
                # A single-bit error cannot be a codeword because every H column
                # is nonzero in the systematic construction.
                self.assertFalse(code.contains(bad))

    def test_mrb(self) -> None:
        rng = np.random.default_rng(2)
        code = systematic_random_code(20, 14, rng, "dense")
        rel = rng.random(code.n)
        gmrb, info = most_reliable_basis_generator(code.G, rel)
        self.assertTrue(np.array_equal(gmrb[:, info], np.eye(code.k, dtype=np.uint8)))
        for _ in range(10):
            u = rng.integers(0, 2, size=code.k, dtype=np.uint8)
            c = (u @ gmrb) & 1
            self.assertTrue(code.contains(c))


class TestQPSK(unittest.TestCase):
    def test_c4_maps(self) -> None:
        words = np.array(list(itertools.product([0, 1], repeat=2)), dtype=np.uint8)
        for bits in words:
            x = bits_to_qpsk(bits)
            for d in range(4):
                out = qpsk_hard_bits(x * (1j ** d))
                recovered = qpsk_hard_bits((x * (1j ** d)) * ((-1j) ** d))
                self.assertTrue(np.array_equal(recovered, bits))
                self.assertEqual(out.size, 2)
            q = qpsk_hard_bits(x * 1j)
            self.assertEqual(int(np.count_nonzero(q ^ bits)), 1)

    def test_stream_cost_identity(self) -> None:
        rng = np.random.default_rng(3)
        bits = rng.integers(0, 2, size=10, dtype=np.uint8)
        path = fixed_slip_path(5, 2, 1)
        sample = simulate_qpsk_awgn(bits, path, 4.0, rng)
        state = one_slip_hypotheses(5, 0.5, [1, 3])
        h = next(s for s in state if np.array_equal(s.path, path))
        hard, costs, base = stream_parameters(sample.y, h, sample.n0)
        for i in range(bits.size):
            flip = hard.copy()
            flip[i] ^= 1
            lhs = h.log_prior + qpsk_loglikelihood(sample.y, flip, h.path, sample.n0)
            self.assertAlmostEqual(lhs, base - costs[i], places=9)


class TestEnumeration(unittest.TestCase):
    def test_subset_sum_order_and_uniqueness(self) -> None:
        costs = np.array([0.7, 0.2, 1.4, 0.2, 2.0])
        e = SubsetSumEnumerator(costs)
        observed = []
        keys = set()
        for _ in range(1 << len(costs)):
            cost, subset = e.pop()
            key = tuple(sorted(int(i) for i in subset))
            self.assertNotIn(key, keys)
            keys.add(key)
            self.assertAlmostEqual(cost, float(costs[list(key)].sum()) if key else 0.0, places=12)
            observed.append(cost)
        self.assertTrue(all(a <= b + 1e-12 for a, b in zip(observed, observed[1:])))
        with self.assertRaises(StopIteration):
            e.pop()


class TestExactDecoder(unittest.TestCase):
    def test_randomized_marginal_ml_agreement(self) -> None:
        for trial in range(8):
            rng = np.random.default_rng(100 + trial)
            code = systematic_random_code(10, 7, rng, "dense" if trial % 2 == 0 else "sparse")
            word = code.encode(rng.integers(0, 2, size=code.k, dtype=np.uint8))
            states = one_slip_hypotheses(5, 0.5, [1, 3])
            true = states[int(rng.integers(len(states)))]
            sample = simulate_qpsk_awgn(word, true.path, float(rng.choice([0.0, 3.0, 6.0])), rng)
            oracle = exhaustive_oracle(sample.y, sample.n0, code, states, max_k=12)
            result = certified_lsgrand(
                sample.y, sample.n0, code, states,
                query_cap=len(states) * (1 << code.n),
                certificate_interval=1,
                oracle_word=np.asarray(oracle.marginal_winner_bits, dtype=np.uint8),
            )
            self.assertTrue(result.certified)
            if not oracle.marginal_tie:
                self.assertEqual(result.decoded_int, oracle.marginal_winner_int)
            direct = direct_probability_scores(sample.y, sample.n0, code, states, max_k=12)
            self.assertEqual(max(direct, key=direct.get), oracle.marginal_winner_int)

    def test_osd_returns_codeword(self) -> None:
        rng = np.random.default_rng(9)
        code = systematic_random_code(24, 18, rng, "dense")
        word = code.encode(rng.integers(0, 2, size=code.k, dtype=np.uint8))
        states = one_slip_hypotheses(12, 0.4, [1, 3])
        sample = simulate_qpsk_awgn(word, states[3].path, 7.0, rng)
        result = latent_osd(sample.y, sample.n0, code, states, order=1, pool_size=8, state_limit=8)
        self.assertTrue(result.success)
        self.assertTrue(code.contains(np.asarray(result.decoded_bits, dtype=np.uint8)))
        self.assertEqual(result.state_codeword_likelihoods, result.complete_marginal_scores * len(states))


class TestDiagnostics(unittest.TestCase):
    def test_rank_bound_and_collisions(self) -> None:
        rng = np.random.default_rng(11)
        code = systematic_random_code(12, 8, rng, "dense")
        word = code.encode(rng.integers(0, 2, size=code.k, dtype=np.uint8))
        path = fixed_slip_path(6, 1, 1)
        rec = rank_separation_record(word, path, len(one_slip_hypotheses(6, 0.5)))
        self.assertEqual(rec["apparent_hamming_weight"], 5)
        self.assertGreater(rec["hard_grand_log2_rank_lower_bound"], rec["latent_family_log2_size"])
        cols = exact_orbit_collision_records(code, canonical_collision_paths(12), max_k=12)
        self.assertTrue(cols)
        self.assertTrue(all(0.0 <= r["collision_fraction"] <= 1.0 for r in cols))


if __name__ == "__main__":
    unittest.main()
