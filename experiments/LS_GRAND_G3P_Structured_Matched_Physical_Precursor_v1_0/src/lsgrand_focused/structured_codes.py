from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from typing import Iterable

import numpy as np

from .gf2 import LinearCode, code_from_generator, permute_code, rank


# ---------------------------------------------------------------------------
# Primitive narrow-sense BCH(63,45,>=7), extended to length 64.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GF2m:
    m: int
    primitive_polynomial: int

    def __post_init__(self) -> None:
        if self.m <= 1:
            raise ValueError("m must exceed one")
        object.__setattr__(self, "order", (1 << self.m) - 1)
        object.__setattr__(self, "mask", (1 << self.m) - 1)
        exp = np.zeros(2 * self.order, dtype=np.int64)
        log = np.full(1 << self.m, -1, dtype=np.int64)
        x = 1
        for i in range(self.order):
            exp[i] = x
            log[x] = i
            x <<= 1
            if x & (1 << self.m):
                x ^= self.primitive_polynomial
            x &= self.mask
        if x != 1 or len(set(exp[: self.order].tolist())) != self.order:
            raise ValueError("polynomial is not primitive for the requested field")
        exp[self.order :] = exp[: self.order]
        object.__setattr__(self, "exp_table", exp)
        object.__setattr__(self, "log_table", log)

    def add(self, a: int, b: int) -> int:
        return int(a) ^ int(b)

    def mul(self, a: int, b: int) -> int:
        a = int(a)
        b = int(b)
        if a == 0 or b == 0:
            return 0
        return int(self.exp_table[int(self.log_table[a]) + int(self.log_table[b])])

    def inv(self, a: int) -> int:
        a = int(a)
        if a == 0:
            raise ZeroDivisionError("zero has no multiplicative inverse")
        return int(self.exp_table[self.order - int(self.log_table[a])])

    def div(self, a: int, b: int) -> int:
        if int(a) == 0:
            return 0
        return self.mul(int(a), self.inv(int(b)))

    def pow_alpha(self, exponent: int) -> int:
        return int(self.exp_table[int(exponent) % self.order])


def _poly_mul_field(a: list[int], b: list[int], field: GF2m) -> list[int]:
    out = [0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        if ai == 0:
            continue
        for j, bj in enumerate(b):
            if bj:
                out[i + j] ^= field.mul(ai, bj)
    return out


def _poly_mul_binary(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = np.asarray(a, dtype=np.uint8).reshape(-1) & 1
    bb = np.asarray(b, dtype=np.uint8).reshape(-1) & 1
    out = np.zeros(aa.size + bb.size - 1, dtype=np.uint8)
    for i in np.flatnonzero(aa):
        out[i : i + bb.size] ^= bb
    return out


def cyclotomic_coset(seed: int, n: int) -> tuple[int, ...]:
    seen: list[int] = []
    x = int(seed) % int(n)
    while x not in seen:
        seen.append(x)
        x = (2 * x) % n
    return tuple(seen)


def bch_generator_polynomial(m: int = 6, designed_distance: int = 7) -> np.ndarray:
    if m != 6:
        raise NotImplementedError("the frozen gate uses the audited GF(64) construction")
    field = GF2m(m=6, primitive_polynomial=0b1000011)  # x^6 + x + 1
    n = field.order
    used: set[tuple[int, ...]] = set()
    minimal_polynomials: list[np.ndarray] = []
    for exponent in range(1, int(designed_distance)):
        coset = cyclotomic_coset(exponent, n)
        canonical = tuple(sorted(coset))
        if canonical in used:
            continue
        used.add(canonical)
        poly = [1]
        for e in coset:
            poly = _poly_mul_field(poly, [field.pow_alpha(e), 1], field)
        if any(c not in (0, 1) for c in poly):
            raise AssertionError(f"minimal polynomial coefficients escaped GF(2): {poly}")
        minimal_polynomials.append(np.asarray(poly, dtype=np.uint8))
    generator = np.array([1], dtype=np.uint8)
    for minimal in minimal_polynomials:
        generator = _poly_mul_binary(generator, minimal)
    return generator


def extended_bch_64_45() -> LinearCode:
    gpoly = bch_generator_polynomial(6, 7)
    n0 = 63
    k = n0 - (gpoly.size - 1)
    if k != 45:
        raise AssertionError(f"unexpected BCH dimension {k}")
    g = np.zeros((k, n0), dtype=np.uint8)
    for i in range(k):
        g[i, i : i + gpoly.size] = gpoly
    parity = (g.sum(axis=1) & 1).reshape(-1, 1).astype(np.uint8)
    gext = np.concatenate([g, parity], axis=1)
    code = code_from_generator(
        name="extended_bch_64_45",
        g=gext,
        family="extended_bch",
        metadata={
            "base_length": 63,
            "designed_distance": 7,
            "correction_radius": 3,
            "primitive_polynomial": "x^6+x+1",
            "generator_polynomial_low_first": gpoly.tolist(),
        },
    )
    if code.n != 64 or code.k != 45:
        raise AssertionError("extended BCH dimensions are incorrect")
    return code


class ExtendedBCHDecoder:
    """Algebraic hard-decision decoder for the frozen extended BCH(64,45) code."""

    def __init__(self, code: LinearCode):
        if code.metadata.get("base_code"):
            self.permutation = np.asarray(code.metadata["permutation"], dtype=int)
            self.inverse_permutation = np.asarray(code.metadata["inverse_permutation"], dtype=int)
            base = extended_bch_64_45()
        else:
            self.permutation = np.arange(code.n, dtype=int)
            self.inverse_permutation = np.arange(code.n, dtype=int)
            base = code
        if base.name != "extended_bch_64_45":
            raise ValueError("ExtendedBCHDecoder requires the frozen eBCH(64,45) code")
        self.code = code
        self.base_code = base
        self.field = GF2m(m=6, primitive_polynomial=0b1000011)
        self.t = 3

    def _syndromes(self, word63: np.ndarray) -> list[int]:
        one_positions = np.flatnonzero(np.asarray(word63, dtype=np.uint8) & 1)
        out: list[int] = []
        for j in range(1, 2 * self.t + 1):
            value = 0
            for pos in one_positions:
                value ^= self.field.pow_alpha(j * int(pos))
            out.append(value)
        return out

    def _berlekamp_massey(self, synd: list[int]) -> tuple[list[int], int]:
        nsynd = len(synd)
        c = [0] * (nsynd + 1)
        b = [0] * (nsynd + 1)
        c[0] = 1
        b[0] = 1
        L = 0
        shift = 1
        beta = 1
        for n in range(nsynd):
            discrepancy = synd[n]
            for i in range(1, L + 1):
                if c[i] and synd[n - i]:
                    discrepancy ^= self.field.mul(c[i], synd[n - i])
            if discrepancy == 0:
                shift += 1
                continue
            previous = c.copy()
            scale = self.field.div(discrepancy, beta)
            for j in range(0, nsynd + 1 - shift):
                if b[j]:
                    c[j + shift] ^= self.field.mul(scale, b[j])
            if 2 * L <= n:
                L = n + 1 - L
                b = previous
                beta = discrepancy
                shift = 1
            else:
                shift += 1
        return c[: L + 1], L

    def _evaluate(self, poly: list[int], x: int) -> int:
        value = 0
        power = 1
        for coeff in poly:
            if coeff:
                value ^= self.field.mul(coeff, power)
            power = self.field.mul(power, x)
        return value

    def decode_base_order(self, hard64: np.ndarray) -> np.ndarray | None:
        r = np.asarray(hard64, dtype=np.uint8).reshape(-1) & 1
        if r.size != 64:
            raise ValueError("expected 64 hard bits")
        base = r[:63].copy()
        synd = self._syndromes(base)
        if any(synd):
            locator, degree = self._berlekamp_massey(synd)
            if degree <= 0 or degree > self.t:
                return None
            errors: list[int] = []
            for pos in range(63):
                x = self.field.pow_alpha(-pos)
                if self._evaluate(locator, x) == 0:
                    errors.append(pos)
            if len(errors) != degree:
                return None
            base[np.asarray(errors, dtype=int)] ^= 1
            if any(self._syndromes(base)):
                return None
        out = np.empty(64, dtype=np.uint8)
        out[:63] = base
        out[63] = int(base.sum() & 1)
        if not self.base_code.contains(out):
            return None
        return out

    def decode(self, hard_permuted: np.ndarray) -> np.ndarray | None:
        r = np.asarray(hard_permuted, dtype=np.uint8).reshape(-1) & 1
        if r.size != 64:
            raise ValueError("expected 64 hard bits")
        base_order = r[self.inverse_permutation]
        decoded = self.decode_base_order(base_order)
        if decoded is None:
            return None
        candidate = decoded[self.permutation]
        return candidate if self.code.contains(candidate) else None


# ---------------------------------------------------------------------------
# Polar-like length-64 code from exact BEC Bhattacharyya recursion.
# ---------------------------------------------------------------------------


def _bit_reverse(i: int, bits: int) -> int:
    out = 0
    for _ in range(bits):
        out = (out << 1) | (i & 1)
        i >>= 1
    return out


def polar_bec_code(n: int = 64, k: int = 48, epsilon: float = 0.5) -> LinearCode:
    if n <= 1 or n & (n - 1):
        raise ValueError("polar length must be a power of two")
    if not (1 <= k < n):
        raise ValueError("invalid polar dimension")
    # The recursion order is kept identical to the F^{\otimes m} SC tree.
    z = np.array([float(epsilon)], dtype=float)
    while z.size < n:
        z = np.concatenate([2.0 * z - z * z, z * z])
    kernel = np.array([[1, 0], [1, 1]], dtype=np.uint8)
    transform = np.array([[1]], dtype=np.uint8)
    for _ in range(int(np.log2(n))):
        transform = np.kron(transform, kernel).astype(np.uint8) & 1
    info = np.argsort(z, kind="stable")[:k]
    g = transform[info]
    if rank(g) != k:
        raise AssertionError("selected polar generator rows are not independent")
    return code_from_generator(
        name=f"polar_bec_{n}_{k}",
        g=g,
        family="polar",
        metadata={
            "construction": "BEC Bhattacharyya recursion in SC-tree order",
            "epsilon": float(epsilon),
            "information_indices": info.tolist(),
        },
    )


class PolarSCDecoder:
    """SC and first-order SC-Flip decoder for the frozen polar code."""

    def __init__(self, code: LinearCode):
        if code.metadata.get("base_code"):
            self.permutation = np.asarray(code.metadata["permutation"], dtype=int)
            self.inverse_permutation = np.asarray(code.metadata["inverse_permutation"], dtype=int)
            base_name = str(code.metadata["base_code"])
            if not base_name.startswith("polar_bec_"):
                raise ValueError("interleaved code is not the frozen polar family")
            base = polar_bec_code(code.n, code.k, float(code.metadata.get("epsilon", 0.5)))
        else:
            self.permutation = np.arange(code.n, dtype=int)
            self.inverse_permutation = np.arange(code.n, dtype=int)
            base = code
        if base.family != "polar":
            raise ValueError("PolarSCDecoder requires a polar code")
        self.code = code
        self.base_code = base
        self.info = np.asarray(base.metadata["information_indices"], dtype=int)
        self.info_set = set(int(x) for x in self.info)
        self.frozen = np.ones(base.n, dtype=bool)
        self.frozen[self.info] = False

    @staticmethod
    def _f(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        # Exact box-plus in a stable log-domain form.
        return np.logaddexp(0.0, a + b) - np.logaddexp(a, b)

    @staticmethod
    def _g(a: np.ndarray, b: np.ndarray, beta: np.ndarray) -> np.ndarray:
        return b + (1.0 - 2.0 * beta.astype(float)) * a

    def _decode_node(
        self,
        alpha: np.ndarray,
        offset: int,
        forced: dict[int, int],
        leaf_llrs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        n = alpha.size
        if n == 1:
            leaf_llrs[offset] = float(alpha[0])
            if self.frozen[offset]:
                bit = 0
            elif offset in forced:
                bit = int(forced[offset]) & 1
            else:
                bit = int(alpha[0] < 0)
            u = np.asarray([bit], dtype=np.uint8)
            return u, u.copy()
        half = n // 2
        left_alpha = self._f(alpha[:half], alpha[half:])
        u_left, beta_left = self._decode_node(left_alpha, offset, forced, leaf_llrs)
        right_alpha = self._g(alpha[:half], alpha[half:], beta_left)
        u_right, beta_right = self._decode_node(right_alpha, offset + half, forced, leaf_llrs)
        u = np.concatenate([u_left, u_right])
        beta = np.concatenate([beta_left ^ beta_right, beta_right])
        return u, beta

    def decode_base_llr(self, llr_base: np.ndarray, forced: dict[int, int] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        alpha = np.asarray(llr_base, dtype=float).reshape(-1)
        if alpha.size != self.base_code.n:
            raise ValueError("polar LLR length mismatch")
        leaf = np.zeros(alpha.size, dtype=float)
        u, codeword = self._decode_node(alpha, 0, {} if forced is None else forced, leaf)
        return u, codeword, leaf

    def decode_candidates(self, llr_permuted: np.ndarray, flip_trials: int = 4) -> list[np.ndarray]:
        llr_new = np.asarray(llr_permuted, dtype=float).reshape(-1)
        llr_base = llr_new[self.inverse_permutation]
        u, word_base, leaf = self.decode_base_llr(llr_base)
        candidates: dict[bytes, np.ndarray] = {}
        word_new = word_base[self.permutation]
        if self.code.contains(word_new):
            candidates[np.packbits(word_new, bitorder="big").tobytes()] = word_new
        order = self.info[np.argsort(np.abs(leaf[self.info]), kind="stable")[: min(int(flip_trials), self.info.size)]]
        for index in order:
            forced = {int(index): int(u[int(index)] ^ 1)}
            _, candidate_base, _ = self.decode_base_llr(llr_base, forced)
            candidate_new = candidate_base[self.permutation]
            if self.code.contains(candidate_new):
                candidates.setdefault(np.packbits(candidate_new, bitorder="big").tobytes(), candidate_new)
        return list(candidates.values())

def build_frozen_code(name: str, rng: np.random.Generator) -> LinearCode:
    key = str(name).lower()
    if key in {"bch", "extended_bch", "extended_bch_64_45"}:
        return extended_bch_64_45()
    if key in {"polar", "polar_64_48"}:
        return polar_bec_code(64, 48, 0.5)
    if key in {"random", "random_64_48"}:
        from .gf2 import systematic_random_code
        return systematic_random_code(64, 48, rng, family="random_control", density=0.5)
    raise KeyError(f"unknown frozen code specification: {name}")
