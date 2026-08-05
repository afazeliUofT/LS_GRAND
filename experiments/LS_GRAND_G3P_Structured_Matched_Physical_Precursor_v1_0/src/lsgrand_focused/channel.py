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
        object.__setattr__(self, "path", np.asarray(self.path, dtype=np.int8).reshape(-1) % 4)


@dataclass(frozen=True)
class ChannelSample:
    y: np.ndarray
    transmitted_bits: np.ndarray
    true_path: np.ndarray
    n0: float
    snr_db: float
    residual_phase: np.ndarray
    label: str


def bits_to_qpsk(bits: np.ndarray) -> np.ndarray:
    b = np.asarray(bits, dtype=np.uint8).reshape(-1) & 1
    if b.size % 2:
        raise ValueError("QPSK requires an even bit count")
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
    p = np.asarray(state_path, dtype=np.int8).reshape(-1) % 4
    if x.size != p.size:
        raise ValueError("symbol/state length mismatch")
    return x * np.power(1j, p)


def compensate_state(y: np.ndarray, state_path: np.ndarray) -> np.ndarray:
    yy = np.asarray(y, dtype=np.complex128).reshape(-1)
    p = np.asarray(state_path, dtype=np.int8).reshape(-1) % 4
    if yy.size != p.size:
        raise ValueError("observation/state length mismatch")
    return yy * np.power(-1j, p)


def qpsk_bit_llrs(compensated_y: np.ndarray, n0: float) -> np.ndarray:
    if n0 <= 0:
        raise ValueError("n0 must be positive")
    z = np.asarray(compensated_y, dtype=np.complex128).reshape(-1)
    out = np.empty(2 * z.size, dtype=float)
    scale = 2.0 * SQRT2 / n0
    out[0::2] = scale * z.real
    out[1::2] = scale * z.imag
    return out


def qpsk_loglikelihood(y: np.ndarray, bits: np.ndarray, state_path: np.ndarray, n0: float) -> float:
    yy = np.asarray(y, dtype=np.complex128).reshape(-1)
    mean = apply_state(bits_to_qpsk(bits), state_path)
    if yy.size != mean.size:
        raise ValueError("observation/codeword mismatch")
    return float(-yy.size * np.log(np.pi * n0) - np.sum(np.abs(yy - mean) ** 2) / n0)


def stream_parameters(y: np.ndarray, state: StateHypothesis, n0: float) -> tuple[np.ndarray, np.ndarray, float]:
    z = compensate_state(y, state.path)
    llr = qpsk_bit_llrs(z, n0)
    hard = (llr < 0).astype(np.uint8)
    base = state.log_prior + qpsk_loglikelihood(y, hard, state.path, n0)
    return hard, np.abs(llr), float(base)


def fixed_slip_path(symbols: int, tau: int, direction: int) -> np.ndarray:
    if not (1 <= int(tau) < int(symbols)):
        raise ValueError("tau must be in [1,symbols-1]")
    p = np.zeros(int(symbols), dtype=np.int8)
    p[int(tau) :] = int(direction) % 4
    return p


def one_slip_hypotheses(
    symbols: int,
    conditional_slip_prior: float = 0.5,
    directions: Sequence[int] = (1, 3),
) -> list[StateHypothesis]:
    if symbols < 2:
        raise ValueError("at least two symbols are required")
    q = float(conditional_slip_prior)
    if not 0.0 <= q <= 1.0:
        raise ValueError("conditional slip prior must be in [0,1]")
    dirs = tuple(int(d) % 4 for d in directions if int(d) % 4)
    if not dirs:
        raise ValueError("at least one nonidentity direction is required")
    hypotheses: list[StateHypothesis] = []
    if q < 1.0:
        hypotheses.append(StateHypothesis(
            index=0,
            path=np.zeros(symbols, dtype=np.int8),
            log_prior=float(np.log(max(1.0 - q, np.finfo(float).tiny))),
            label="no_slip",
            slip_count=0,
            slip_locations=(),
            increments=(),
        ))
    locations = range(1, symbols)
    if q > 0:
        mass = q / ((symbols - 1) * len(dirs))
        for tau in locations:
            for direction in dirs:
                hypotheses.append(StateHypothesis(
                    index=len(hypotheses),
                    path=fixed_slip_path(symbols, tau, direction),
                    log_prior=float(np.log(mass)),
                    label=f"slip_tau{tau}_d{direction}",
                    slip_count=1,
                    slip_locations=(tau,),
                    increments=(direction,),
                ))
    lps = np.asarray([h.log_prior for h in hypotheses], dtype=float)
    m = float(lps.max())
    z = m + float(np.log(np.exp(lps - m).sum()))
    return [StateHypothesis(
        index=i,
        path=h.path,
        log_prior=float(h.log_prior - z),
        label=h.label,
        slip_count=h.slip_count,
        slip_locations=h.slip_locations,
        increments=h.increments,
    ) for i, h in enumerate(hypotheses)]


def sample_slip_path(
    symbols: int,
    rng: np.random.Generator,
    *,
    slip: bool,
    directions: Sequence[int] = (1, 3),
    location_mode: str = "uniform",
    location_fraction: float | None = None,
) -> tuple[np.ndarray, str]:
    if not slip:
        return np.zeros(symbols, dtype=np.int8), "no_slip"
    dirs = tuple(int(d) % 4 for d in directions if int(d) % 4)
    if location_fraction is not None:
        tau = min(max(1, int(round(float(location_fraction) * symbols))), symbols - 1)
    elif location_mode == "early":
        lo, hi = 1, max(2, int(np.ceil(0.35 * symbols)))
        tau = int(rng.integers(lo, hi))
    elif location_mode == "middle":
        lo = max(1, int(np.floor(0.35 * symbols)))
        hi = min(symbols, max(lo + 1, int(np.ceil(0.65 * symbols))))
        tau = int(rng.integers(lo, hi))
    elif location_mode == "late":
        lo = max(1, int(np.floor(0.65 * symbols)))
        tau = int(rng.integers(lo, symbols))
    elif location_mode == "uniform":
        tau = int(rng.integers(1, symbols))
    else:
        raise ValueError(f"unknown location mode: {location_mode}")
    direction = int(rng.choice(dirs))
    return fixed_slip_path(symbols, tau, direction), f"slip_tau{tau}_d{direction}"


def simulate_residual_slip_qpsk(
    bits: np.ndarray,
    state_path: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
    *,
    initial_phase_std_deg: float = 1.0,
    innovation_phase_std_deg: float = 0.35,
    frequency_offset_std_deg_per_symbol: float = 0.0,
    label: str = "",
) -> ChannelSample:
    x = bits_to_qpsk(bits)
    path = np.asarray(state_path, dtype=np.int8).reshape(-1)
    if x.size != path.size:
        raise ValueError("state path length mismatch")
    init = np.deg2rad(float(initial_phase_std_deg)) * float(rng.standard_normal())
    innovation = np.deg2rad(float(innovation_phase_std_deg)) * rng.standard_normal(x.size)
    wiener = init + np.cumsum(innovation)
    freq = np.deg2rad(float(frequency_offset_std_deg_per_symbol)) * float(rng.standard_normal())
    residual = wiener + freq * np.arange(x.size, dtype=float)
    mean = apply_state(x, path) * np.exp(1j * residual)
    n0 = float(10.0 ** (-float(snr_db) / 10.0))
    sigma = float(np.sqrt(n0 / 2.0))
    noise = sigma * (rng.standard_normal(x.size) + 1j * rng.standard_normal(x.size))
    y = mean + noise
    return ChannelSample(
        y=y,
        transmitted_bits=np.asarray(bits, dtype=np.uint8).copy(),
        true_path=path.copy(),
        n0=n0,
        snr_db=float(snr_db),
        residual_phase=residual,
        label=label,
    )
