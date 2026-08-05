from __future__ import annotations

import itertools
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from lsgrand_focused.affine import affine_map_for_path, linear_code_affine_collision
from lsgrand_focused.campaign import adjudicate, load_config, load_g0_chain_lock, load_novelty_matrix, normal_code_decoder
from lsgrand_focused.channel import (
    apply_state,
    bits_to_qpsk,
    fixed_slip_path,
    one_slip_hypotheses,
    qpsk_hard_bits,
    qpsk_loglikelihood,
    simulate_residual_slip_qpsk,
)
from lsgrand_focused.decoders import batch_marginal_scores, latent_list_decode, latent_osd_batch
from lsgrand_focused.gf2 import permute_code, systematic_random_code
from lsgrand_focused.search import SubsetSumEnumerator, logsumexp
from lsgrand_focused.structured_codes import ExtendedBCHDecoder, PolarSCDecoder, extended_bch_64_45, polar_bec_code


class TestStructuredCodes(unittest.TestCase):
    def test_bch_dimensions_and_decoder(self) -> None:
        code = extended_bch_64_45()
        self.assertEqual((code.n, code.k), (64, 45))
        decoder = ExtendedBCHDecoder(code)
        rng = np.random.default_rng(11)
        for weight in range(4):
            for _ in range(10):
                word = code.encode(rng.integers(0, 2, size=code.k, dtype=np.uint8))
                received = word.copy()
                if weight:
                    received[rng.choice(code.n, size=weight, replace=False)] ^= 1
                decoded = decoder.decode(received)
                self.assertIsNotNone(decoded)
                self.assertTrue(np.array_equal(decoded, word))

    def test_bch_interleaved_decoder(self) -> None:
        code = extended_bch_64_45()
        rng = np.random.default_rng(12)
        permutation = rng.permutation(code.n)
        interleaved = permute_code(code, permutation)
        decoder = ExtendedBCHDecoder(interleaved)
        word = interleaved.encode(rng.integers(0, 2, size=interleaved.k, dtype=np.uint8))
        received = word.copy()
        received[rng.choice(code.n, size=3, replace=False)] ^= 1
        decoded = decoder.decode(received)
        self.assertIsNotNone(decoded)
        self.assertTrue(np.array_equal(decoded, word))

    def test_polar_code(self) -> None:
        code = polar_bec_code(64, 48)
        rng = np.random.default_rng(13)
        for _ in range(8):
            word = code.encode(rng.integers(0, 2, size=code.k, dtype=np.uint8))
            self.assertTrue(code.contains(word))


    def test_frozen_normal_decoders_noiseless(self) -> None:
        root = Path(__file__).resolve().parents[1]
        decoder_cfg = load_config(root / "configs" / "gate.json")["decoders"]
        rng = np.random.default_rng(131)
        for code in [extended_bch_64_45(), polar_bec_code(64, 48)]:
            word = code.encode(rng.integers(0, 2, size=code.k, dtype=np.uint8))
            y = bits_to_qpsk(word)
            result = normal_code_decoder(y, 0.01, code, decoder_cfg)
            self.assertTrue(result.success)
            self.assertTrue(np.array_equal(np.asarray(result.decoded_bits, dtype=np.uint8), word))

    def test_polar_sc_noiseless(self) -> None:
        code = polar_bec_code(64, 48)
        decoder = PolarSCDecoder(code)
        rng = np.random.default_rng(130)
        for _ in range(8):
            word = code.encode(rng.integers(0, 2, size=code.k, dtype=np.uint8))
            llr = np.where(word == 0, 50.0, -50.0)
            candidates = decoder.decode_candidates(llr, flip_trials=2)
            self.assertTrue(any(np.array_equal(candidate, word) for candidate in candidates))


class TestSearchAndMetrics(unittest.TestCase):
    def test_subset_enumerator_exact(self) -> None:
        costs = np.array([0.2, 1.1, 0.5, 2.0])
        enum = SubsetSumEnumerator(costs)
        observed = [enum.pop()[0] for _ in range(1 << costs.size)]
        expected = sorted(sum(costs[i] for i in range(costs.size) if mask & (1 << i)) for mask in range(1 << costs.size))
        self.assertTrue(np.allclose(observed, expected))

    def test_batch_marginal_matches_direct(self) -> None:
        rng = np.random.default_rng(14)
        code = systematic_random_code(8, 4, rng)
        word = code.encode(rng.integers(0, 2, size=code.k, dtype=np.uint8))
        states = one_slip_hypotheses(4, 0.4, (1, 3))
        sample = simulate_residual_slip_qpsk(word, states[0].path, 6.0, rng, initial_phase_std_deg=0.0, innovation_phase_std_deg=0.0)
        cands = code.enumerate_codewords(max_k=8)
        scores, _, _, _ = batch_marginal_scores(sample.y, sample.n0, cands, states)
        direct = []
        for candidate in cands:
            direct.append(logsumexp(s.log_prior + qpsk_loglikelihood(sample.y, candidate, s.path, sample.n0) for s in states))
        self.assertTrue(np.allclose(scores, direct, atol=1e-10))

    def test_list_and_osd_return_codewords(self) -> None:
        rng = np.random.default_rng(15)
        code = systematic_random_code(16, 8, rng)
        word = code.encode(rng.integers(0, 2, size=code.k, dtype=np.uint8))
        path = fixed_slip_path(8, 2, 1)
        sample = simulate_residual_slip_qpsk(word, path, 8.0, rng, initial_phase_std_deg=0.0, innovation_phase_std_deg=0.0)
        states = one_slip_hypotheses(8, 0.5, (1, 3))
        ls = latent_list_decode(sample.y, sample.n0, code, states, list_size=1, query_cap=20000, marginal_rescore=False)
        osd = latent_osd_batch(sample.y, sample.n0, code, states, order=2, pool_size=6)
        self.assertTrue(ls.success and code.contains(np.asarray(ls.decoded_bits, dtype=np.uint8)))
        self.assertTrue(osd.success and code.contains(np.asarray(osd.decoded_bits, dtype=np.uint8)))


class TestAffine(unittest.TestCase):
    def test_affine_waveform_map(self) -> None:
        rng = np.random.default_rng(16)
        bits = rng.integers(0, 2, size=16, dtype=np.uint8)
        path = fixed_slip_path(8, 3, 3)
        a, b = affine_map_for_path(path)
        algebra = ((a @ bits) ^ b) & 1
        waveform = qpsk_hard_bits(apply_state(bits_to_qpsk(bits), path))
        self.assertTrue(np.array_equal(algebra, waveform))

    def test_collision_matches_enumeration(self) -> None:
        rng = np.random.default_rng(17)
        code = systematic_random_code(8, 4, rng)
        path = fixed_slip_path(4, 2, 1)
        result = linear_code_affine_collision(code, path)
        a, b = affine_map_for_path(path)
        cws = code.enumerate_codewords(max_k=8)
        hits = sum(code.contains(((a @ c) ^ b) & 1) for c in cws)
        self.assertAlmostEqual(result.collision_fraction, hits / len(cws))


class TestConfig(unittest.TestCase):
    def test_configs_load(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for name in ["smoke.json", "gate.json", "stress.json"]:
            cfg = load_config(root / "configs" / name)
            self.assertIn("thresholds", cfg)
        gate = load_config(root / "configs" / "gate.json")
        self.assertLessEqual(gate["detector"]["target_false_alarm_rate"], 0.001)
        self.assertGreaterEqual(gate["thresholds"]["physical_tail_quantile"], 0.999)
        self.assertGreaterEqual(gate["thresholds"]["min_eligible_work_pairs"], 100)

    def test_physical_candidate_must_pass_all_prior_gates(self) -> None:
        root = Path(__file__).resolve().parents[1]
        cfg = load_config(root / "configs" / "gate.json")
        chain = load_g0_chain_lock(root)
        novelty = load_novelty_matrix(root)
        orbit = pd.DataFrame([
            {"code_id": "BCH64_45", "family": "extended_bch", "orbit_safe": True},
            {"code_id": "POLAR64_48", "family": "polar", "orbit_safe": True},
        ])
        detector = pd.DataFrame([
            {"code_id": "BCH64_45", "alarm_rate_no_slip": 0.0, "alarm_rate_slip_early": 1.0, "alarm_rate_slip_uniform": 1.0},
            {"code_id": "POLAR64_48", "alarm_rate_no_slip": 0.0, "alarm_rate_slip_early": 1.0, "alarm_rate_slip_uniform": 1.0},
        ])
        rows = []
        for code_id, family, specific in [
            ("BCH64_45", "extended_bch", "event_state_sweep_chase_bch"),
            ("POLAR64_48", "polar", "event_state_sweep_polar_scflip"),
        ]:
            common = {
                "code_id": code_id, "family": family, "paired_trials": 200,
                "eligible_work_pairs": 200, "candidate_bler": 0.01,
                "candidate_bler_high": 0.02, "candidate_cap_rate": 0.0,
                "candidate_minus_baseline_error_rate": 0.0,
                "error_diff_low": 0.0, "error_diff_high": 0.0,
                "baseline_to_candidate_wall_median_ratio": 10.0,
                "wall_ratio_low": 5.0, "wall_ratio_high": 15.0,
                "baseline_to_candidate_component_median_ratio": 5.0,
                "component_ratio_low": 2.0, "component_ratio_high": 8.0,
                "baseline_to_candidate_p99_wall_ratio": 5.0, "mcnemar": "{}",
            }
            rows.append({**common, "candidate": "event_ls_first", "baseline": "event_state_sweep_osd"})
            rows.append({**common, "candidate": "event_ls_adaptive_l2", "baseline": specific})
        comparisons = pd.DataFrame(rows)
        overhead = pd.DataFrame([
            {"code_id": code_id, "decoder": decoder, "false_alarm_rate": 0.0,
             "median_wall_overhead_ratio": 1.01, "p99_wall_overhead_ratio": 1.02}
            for code_id in ["BCH64_45", "POLAR64_48"]
            for decoder in ["event_ls_first", "event_ls_adaptive_l2"]
        ])
        mixture_rows = []
        for code_id, baseline in [
            ("BCH64_45", "event_state_sweep_chase_bch"),
            ("POLAR64_48", "event_state_sweep_polar_scflip"),
        ]:
            for decoder, tail in [(baseline, 10.0), ("event_ls_first", 1.0), ("event_ls_adaptive_l2", 1.0)]:
                mixture_rows.append({
                    "code_id": code_id, "decoder": decoder,
                    "physical_slip_probability": cfg["thresholds"]["physical_gate_slip_probability"],
                    "p99_wall_seconds": tail, "p999_wall_seconds": tail,
                    "unconditional_bler": 0.0,
                })
        verdict, _ = adjudicate(
            cfg, chain, novelty, orbit, detector, comparisons, overhead,
            pd.DataFrame(mixture_rows), True,
        )
        gate_map = {g["gate"]: g for g in verdict["gates"]}
        self.assertTrue(gate_map["C3_APPROXIMATE_STRUCTURED_CODE_FRONTIER"]["pass"])
        self.assertTrue(gate_map["C4_CODE_SPECIFIC_BASELINES"]["pass"])
        self.assertFalse(gate_map["C6_SYNTHETIC_PHYSICAL_MIXTURE_SCREEN"]["pass"])
        self.assertEqual(verdict["verdict"], "CONTINUE_CONDITIONAL_ON_EXTERNAL_ALARM_ONLY")

    def test_g0_chain_lock_blocks_broad_claims(self) -> None:
        root = Path(__file__).resolve().parents[1]
        chain = load_g0_chain_lock(root)
        claims = load_novelty_matrix(root)
        self.assertEqual(chain["source_commit"], "0d2866b091576fe07521172378ded33da79ed545")
        disp = dict(zip(claims["claim_id"].astype(str), claims["disposition"].astype(str)))
        for cid in ["C01", "C02", "C03", "C04", "C10", "C11", "C12"]:
            self.assertTrue(disp[cid].startswith("BLOCK"))
        for cid in ["C05", "C06", "C07", "C08"]:
            self.assertTrue(disp[cid].startswith("ALLOW"))
        self.assertEqual(chain["field_defining_program"], "HOLD_NOT_AUTHORIZED")


if __name__ == "__main__":
    unittest.main()
