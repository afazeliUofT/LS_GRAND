from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from lsgrand_g4.affine import affine_map_for_path, linear_code_affine_collision
from lsgrand_g4.campaign import load_chain, load_config, normal_code_decoder
from lsgrand_g4.channel import apply_state, bits_to_qpsk, fixed_slip_path, one_slip_hypotheses, qpsk_hard_bits
from lsgrand_g4.decoders import (
    dqpsk_encode,
    dqpsk_observations,
    latent_list_decode,
    latent_osd_batch,
    posterior_pruned_bch_sweep,
    posterior_pruned_polar_sweep,
    state_marginal_code_specific,
)
from lsgrand_g4.gf2 import permute_code, systematic_random_code
from lsgrand_g4.search import SubsetSumEnumerator
from lsgrand_g4.structured_codes import ExtendedBCHDecoder, PolarSCDecoder, extended_bch_64_45, polar_bec_code
from lsgrand_g4.trace import generate_vv_trace_batch


class TestStructuredCodes(unittest.TestCase):
    def test_bch_dimensions_and_decoder(self) -> None:
        code = extended_bch_64_45()
        self.assertEqual((code.n, code.k), (64, 45))
        decoder = ExtendedBCHDecoder(code)
        rng = np.random.default_rng(11)
        for weight in range(4):
            word = code.encode(rng.integers(0, 2, size=code.k, dtype=np.uint8))
            received = word.copy()
            if weight:
                received[rng.choice(code.n, size=weight, replace=False)] ^= 1
            self.assertTrue(np.array_equal(decoder.decode(received), word))

    def test_bch_interleaved_decoder(self) -> None:
        code = extended_bch_64_45()
        rng = np.random.default_rng(12)
        interleaved = permute_code(code, rng.permutation(code.n))
        decoder = ExtendedBCHDecoder(interleaved)
        word = interleaved.encode(rng.integers(0, 2, size=interleaved.k, dtype=np.uint8))
        received = word.copy()
        received[rng.choice(code.n, size=3, replace=False)] ^= 1
        self.assertTrue(np.array_equal(decoder.decode(received), word))

    def test_polar_noiseless(self) -> None:
        code = polar_bec_code(64, 48)
        decoder = PolarSCDecoder(code)
        rng = np.random.default_rng(13)
        word = code.encode(rng.integers(0, 2, size=code.k, dtype=np.uint8))
        llr = np.where(word == 0, 50.0, -50.0)
        self.assertTrue(any(np.array_equal(c, word) for c in decoder.decode_candidates(llr, flip_trials=4)))

    def test_normal_decoders_noiseless(self) -> None:
        root = Path(__file__).resolve().parents[1]
        dcfg = load_config(root / "configs" / "gate.json")["decoders"]
        rng = np.random.default_rng(14)
        for code in [extended_bch_64_45(), polar_bec_code(64, 48)]:
            word = code.encode(rng.integers(0, 2, size=code.k, dtype=np.uint8))
            result = normal_code_decoder(bits_to_qpsk(word), 0.01, code, dcfg)
            self.assertTrue(result.success)
            self.assertTrue(np.array_equal(np.asarray(result.decoded_bits, dtype=np.uint8), word))


class TestTraceAndSearch(unittest.TestCase):
    def test_trace_shapes_and_zero_linewidth(self) -> None:
        rng = np.random.default_rng(20)
        batch = generate_vv_trace_batch(
            rng, frames=64, warmup_symbols=32, payload_symbols=35,
            symbol_rate_baud=28e9, combined_linewidth_hz=0.0,
            snr_db=30.0, vv_window=9,
        )
        self.assertEqual(batch.gain.shape, (64, 35))
        self.assertTrue(np.all(batch.event_class == 0))

    def test_trace_produces_finite_metrics(self) -> None:
        rng = np.random.default_rng(21)
        batch = generate_vv_trace_batch(
            rng, frames=128, warmup_symbols=32, payload_symbols=35,
            symbol_rate_baud=28e9, combined_linewidth_hz=20e6,
            snr_db=7.0, vv_window=9,
        )
        self.assertTrue(np.all(np.isfinite(batch.gain.real)))
        self.assertTrue(np.all(np.isfinite(batch.cpe_confidence_min)))

    def test_subset_enumerator_exact(self) -> None:
        costs = np.array([0.2, 1.1, 0.5, 2.0])
        enum = SubsetSumEnumerator(costs)
        observed = [enum.pop()[0] for _ in range(1 << costs.size)]
        expected = sorted(sum(costs[i] for i in range(costs.size) if mask & (1 << i)) for mask in range(1 << costs.size))
        self.assertTrue(np.allclose(observed, expected))

    def test_list_osd_and_pruned_baselines_return_codewords(self) -> None:
        rng = np.random.default_rng(22)
        for code in [extended_bch_64_45(), polar_bec_code(64, 48)]:
            word = code.encode(rng.integers(0, 2, size=code.k, dtype=np.uint8))
            path = fixed_slip_path(code.n // 2, 5, 1)
            y = apply_state(bits_to_qpsk(word), path)
            states = one_slip_hypotheses(code.n // 2, 0.9, (1, 3))
            ls = latent_list_decode(y, 0.01, code, states, list_size=1, query_cap=10000, marginal_rescore=False)
            osd = latent_osd_batch(y, 0.01, code, states, order=1, pool_size=4, state_limit=None)
            self.assertTrue(ls.success and code.contains(np.asarray(ls.decoded_bits, dtype=np.uint8)))
            self.assertTrue(osd.success and code.contains(np.asarray(osd.decoded_bits, dtype=np.uint8)))
            if code.family == "extended_bch":
                p = posterior_pruned_bch_sweep(y, 0.01, code, states, word, state_limit=2, chase_order=1, chase_pool=4)
            else:
                p = posterior_pruned_polar_sweep(y, 0.01, code, states, word, state_limit=2, flip_trials=2)
            self.assertTrue(p.success and code.contains(np.asarray(p.decoded_bits, dtype=np.uint8)))
            m = state_marginal_code_specific(
                y, 0.01, code, states,
                bch_chase_order=1, bch_chase_pool=4,
                polar_flip_trials=2, candidate_cap=10000,
            )
            self.assertTrue(m.success and code.contains(np.asarray(m.decoded_bits, dtype=np.uint8)))

    def test_dqpsk_noiseless_roundtrip(self) -> None:
        rng = np.random.default_rng(23)
        bits = rng.integers(0, 2, size=64, dtype=np.uint8)
        tx = dqpsk_encode(bits)
        obs = dqpsk_observations(tx)
        hard = qpsk_hard_bits(obs)
        self.assertTrue(np.array_equal(hard, bits))


class TestAffineAndPolicy(unittest.TestCase):
    def test_affine_waveform_map(self) -> None:
        rng = np.random.default_rng(30)
        bits = rng.integers(0, 2, size=16, dtype=np.uint8)
        path = fixed_slip_path(8, 3, 3)
        a, b = affine_map_for_path(path)
        algebra = ((a @ bits) ^ b) & 1
        waveform = qpsk_hard_bits(apply_state(bits_to_qpsk(bits), path))
        self.assertTrue(np.array_equal(algebra, waveform))

    def test_collision_matches_enumeration(self) -> None:
        rng = np.random.default_rng(31)
        code = systematic_random_code(8, 4, rng)
        path = fixed_slip_path(4, 2, 1)
        result = linear_code_affine_collision(code, path)
        a, b = affine_map_for_path(path)
        cws = code.enumerate_codewords(max_k=8)
        hits = sum(code.contains(((a @ c) ^ b) & 1) for c in cws)
        self.assertAlmostEqual(result.collision_fraction, hits / len(cws))

    def test_configs_and_chain_locks(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for name in ["smoke.json", "gate.json", "stress.json"]:
            cfg = load_config(root / "configs" / name)
            self.assertIn("trace", cfg)
        g0, g3p, claims = load_chain(root)
        self.assertEqual(g0["source_commit"], "0d2866b091576fe07521172378ded33da79ed545")
        self.assertEqual(g3p["source_commit"], "5edb4083ec30d44061f0020a2701fd189f87df23")
        disp = dict(zip(claims["claim_id"].astype(str), claims["disposition"].astype(str)))
        for cid in ["C01", "C02", "C03", "C04", "C10", "C11", "C12"]:
            self.assertTrue(disp[cid].startswith("BLOCK"))
        for cid in ["C05", "C06", "C07", "C08"]:
            self.assertTrue(disp[cid].startswith("ALLOW"))


if __name__ == "__main__":
    unittest.main()
