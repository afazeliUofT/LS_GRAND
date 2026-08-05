from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


SQRT2 = float(np.sqrt(2.0))


@dataclass(frozen=True)
class StateHypothesis:
    index: int
    path: np.ndarray
    log_prior: float
    label: str
    slip_count: int
    slip_locations: tuple[int, ...]
    increments: tuple[int, ...]

    def __post_init__(self) -> None:
        p = np.asarray(self.path, dtype=np.int8).reshape(-1) % 4
        object.__setattr__(self, "path", p)


@dataclass(frozen=True)
class ChannelSample:
    y: np.ndarray
    transmitted_bits: np.ndarray
    symbols: np.ndarray
    true_path: np.ndarray
    n0: float
    snr_db: float
    true_label: str


def bits_to_qpsk(bits: np.ndarray) -> np.ndarray:
    b = np.asarray(bits, dtype=np.uint8).reshape(-1) & 1
    if b.size % 2:
        raise ValueError("QPSK requires an even number of bits")
    i = 1.0 - 2.0 * b[0::2].astype(float)
    q = 1.0 - 2.0 * b[1::2].astype(float)
    return (i + 1j * q) / SQRT2


def qpsk_hard_bits(symbols: np.ndarray) -> np.ndarray:
    z = np.asarray(symbols, dtype=np.complex128).reshape(-1)
    out = np.empty(2 * z.size, dtype=np.uint8)
    out[0::2] = (z.real < 0).astype(np.uint8)
    out[1::2] = (z.imag < 0).astype(np.uint8)
    return out


def apply_state(symbols: np.ndarray, state_path: np.ndarray) -> np.ndarray:
    x = np.asarray(symbols, dtype=np.complex128).reshape(-1)
    s = np.asarray(state_path, dtype=np.int8).reshape(-1) % 4
    if x.size != s.size:
        raise ValueError("symbol/state length mismatch")
    return x * np.power(1j, s)


def compensate_state(y: np.ndarray, state_path: np.ndarray) -> np.ndarray:
    yy = np.asarray(y, dtype=np.complex128).reshape(-1)
    s = np.asarray(state_path, dtype=np.int8).reshape(-1) % 4
    if yy.size != s.size:
        raise ValueError("observation/state length mismatch")
    return yy * np.power(-1j, s)


def qpsk_bit_llrs(compensated_y: np.ndarray, n0: float) -> np.ndarray:
    """Exact bit LLRs for sign-labeled QPSK and CN(0,N0) noise."""
    if n0 <= 0:
        raise ValueError("N0 must be positive")
    z = np.asarray(compensated_y, dtype=np.complex128).reshape(-1)
    llr = np.empty(2 * z.size, dtype=float)
    scale = 2.0 * SQRT2 / n0
    llr[0::2] = scale * z.real
    llr[1::2] = scale * z.imag
    return llr


def qpsk_loglikelihood(y: np.ndarray, bits: np.ndarray, state_path: np.ndarray, n0: float) -> float:
    if n0 <= 0:
        raise ValueError("N0 must be positive")
    yy = np.asarray(y, dtype=np.complex128).reshape(-1)
    mean = apply_state(bits_to_qpsk(bits), state_path)
    if yy.size != mean.size:
        raise ValueError("observation/codeword length mismatch")
    return float(-yy.size * np.log(np.pi * n0) - np.sum(np.abs(yy - mean) ** 2) / n0)


def stream_parameters(y: np.ndarray, state: StateHypothesis, n0: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Hard word, flip costs, and log weight of the hard word for one state."""
    z = compensate_state(y, state.path)
    llr = qpsk_bit_llrs(z, n0)
    hard = (llr < 0).astype(np.uint8)
    base = state.log_prior + qpsk_loglikelihood(y, hard, state.path, n0)
    return hard, np.abs(llr), float(base)


def one_slip_hypotheses(
    symbols: int,
    frame_slip_prob: float,
    directions: Sequence[int] = (1, 2, 3),
    include_no_slip: bool = True,
) -> list[StateHypothesis]:
    """Finite exact family: no slip or one persistent C4 increment.

    A slip at location `tau` affects symbols tau, tau+1, ..., with
    tau in {1,...,symbols-1}; symbol zero anchors the initial phase.
    """
    if symbols < 1:
        raise ValueError("symbols must be positive")
    if not (0.0 <= frame_slip_prob <= 1.0):
        raise ValueError("frame_slip_prob must lie in [0,1]")
    dirs = tuple(int(d) % 4 for d in directions if int(d) % 4 != 0)
    if not dirs and frame_slip_prob > 0:
        raise ValueError("at least one nonzero direction is required")
    out: list[StateHypothesis] = []
    idx = 0
    if include_no_slip and frame_slip_prob < 1.0:
        out.append(
            StateHypothesis(
                index=idx,
                path=np.zeros(symbols, dtype=np.int8),
                log_prior=float(np.log1p(-frame_slip_prob)),
                label="no_slip",
                slip_count=0,
                slip_locations=(),
                increments=(),
            )
        )
        idx += 1
    locations = list(range(1, symbols))
    if frame_slip_prob > 0 and locations:
        mass = frame_slip_prob / (len(locations) * len(dirs))
        lp = float(np.log(mass))
        for tau in locations:
            for d in dirs:
                path = np.zeros(symbols, dtype=np.int8)
                path[tau:] = d
                out.append(
                    StateHypothesis(
                        index=idx,
                        path=path,
                        log_prior=lp,
                        label=f"slip_tau{tau}_d{d}",
                        slip_count=1,
                        slip_locations=(tau,),
                        increments=(d,),
                    )
                )
                idx += 1
    if not out:
        # Degenerate one-symbol or q=0 case.
        out.append(
            StateHypothesis(
                index=0,
                path=np.zeros(symbols, dtype=np.int8),
                log_prior=0.0,
                label="no_slip",
                slip_count=0,
                slip_locations=(),
                increments=(),
            )
        )
    # Normalize to remove roundoff and support q=1 with no no-slip state.
    lps = np.array([h.log_prior for h in out], dtype=float)
    m = float(np.max(lps))
    z = m + float(np.log(np.exp(lps - m).sum()))
    return [
        StateHypothesis(
            index=i,
            path=h.path,
            log_prior=h.log_prior - z,
            label=h.label,
            slip_count=h.slip_count,
            slip_locations=h.slip_locations,
            increments=h.increments,
        )
        for i, h in enumerate(out)
    ]


def fixed_slip_path(symbols: int, tau: int, direction: int) -> np.ndarray:
    if not (1 <= tau < symbols):
        raise ValueError("tau must be in [1,symbols-1]")
    p = np.zeros(symbols, dtype=np.int8)
    p[tau:] = int(direction) % 4
    return p


def two_slip_path(symbols: int, tau1: int, d1: int, tau2: int, d2: int) -> np.ndarray:
    if not (1 <= tau1 < tau2 < symbols):
        raise ValueError("require 1 <= tau1 < tau2 < symbols")
    p = np.zeros(symbols, dtype=np.int8)
    p[tau1:] = int(d1) % 4
    p[tau2:] = (int(d1) + int(d2)) % 4
    return p


def sample_state_path(
    symbols: int,
    rng: np.random.Generator,
    frame_slip_prob: float,
    directions: Sequence[int] = (1, 2, 3),
    forced_location_fraction: float | None = None,
    two_slip_fraction: float = 0.0,
) -> tuple[np.ndarray, str]:
    if symbols < 2 or rng.random() >= frame_slip_prob:
        return np.zeros(symbols, dtype=np.int8), "no_slip"
    dirs = tuple(int(d) % 4 for d in directions if int(d) % 4)
    if two_slip_fraction > 0 and symbols >= 4 and rng.random() < two_slip_fraction:
        tau1 = int(rng.integers(1, symbols - 2))
        tau2 = int(rng.integers(tau1 + 1, symbols))
        d1 = int(rng.choice(dirs))
        d2 = int(rng.choice(dirs))
        return two_slip_path(symbols, tau1, d1, tau2, d2), f"two_slip_{tau1}_{d1}_{tau2}_{d2}"
    if forced_location_fraction is None:
        tau = int(rng.integers(1, symbols))
    else:
        tau = int(round(float(forced_location_fraction) * symbols))
        tau = min(max(1, tau), symbols - 1)
    d = int(rng.choice(dirs))
    return fixed_slip_path(symbols, tau, d), f"slip_{tau}_{d}"


def simulate_qpsk_awgn(
    bits: np.ndarray,
    state_path: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
    true_label: str = "",
) -> ChannelSample:
    x = bits_to_qpsk(bits)
    if x.size != np.asarray(state_path).size:
        raise ValueError("state path length mismatch")
    n0 = float(10.0 ** (-float(snr_db) / 10.0))
    sigma = float(np.sqrt(n0 / 2.0))
    noise = sigma * (rng.standard_normal(x.size) + 1j * rng.standard_normal(x.size))
    y = apply_state(x, state_path) + noise
    return ChannelSample(
        y=y,
        transmitted_bits=np.asarray(bits, dtype=np.uint8).copy(),
        symbols=x,
        true_path=np.asarray(state_path, dtype=np.int8).copy(),
        n0=n0,
        snr_db=float(snr_db),
        true_label=true_label,
    )


def affine_transform_bits(bits: np.ndarray, state_path: np.ndarray) -> np.ndarray:
    """Noiseless hard-label transform induced by a C4 state trajectory."""
    return qpsk_hard_bits(apply_state(bits_to_qpsk(bits), state_path))
