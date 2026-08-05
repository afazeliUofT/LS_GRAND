from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class VVTraceBatch:
    """Carrier-recovery output traces for fixed-size payload windows.

    ``gain`` is the complex multiplicative sequence seen after fourth-power
    Viterbi--Viterbi carrier recovery and initial-branch anchoring.  Applying it
    to any unit-modulus QPSK sequence preserves the residual phase/noise law.
    ``event_class`` is 0=no branch slip, 1=one persistent +/-pi/2 slip, 2=other.
    """

    gain: np.ndarray
    event_class: np.ndarray
    slip_location: np.ndarray
    slip_direction: np.ndarray
    cpe_confidence_min: np.ndarray
    cpe_confidence_mean: np.ndarray
    cpe_estimate_range: np.ndarray


@dataclass(frozen=True)
class TracePool:
    gain_no_slip: np.ndarray
    gain_one_slip: np.ndarray
    gain_other: np.ndarray
    one_slip_locations: np.ndarray
    one_slip_directions: np.ndarray
    summary: dict[str, Any]


def _moving_average_causal(z: np.ndarray, window: int) -> np.ndarray:
    if z.ndim != 2:
        raise ValueError("z must have shape [frames, symbols]")
    w = max(1, int(window))
    c = np.cumsum(z, axis=1, dtype=np.complex128)
    out = c.copy()
    if w < z.shape[1]:
        out[:, w:] = c[:, w:] - c[:, :-w]
    counts = np.minimum(np.arange(1, z.shape[1] + 1), w).astype(float)
    return out / counts[None, :]


def generate_vv_trace_batch(
    rng: np.random.Generator,
    *,
    frames: int,
    warmup_symbols: int,
    payload_symbols: int,
    symbol_rate_baud: float,
    combined_linewidth_hz: float,
    snr_db: float,
    vv_window: int,
) -> VVTraceBatch:
    """Generate a standard phase-noise/Viterbi--Viterbi residual trace batch.

    The laser phase is a Wiener process with per-symbol innovation variance
    ``2*pi*Delta_nu/R_s``.  A causal fourth-power estimator is branch-unwrapped
    by nearest-neighbour continuation.  The estimator's initial quadrant at the
    payload boundary is removed, matching the frozen known-initial-state model.
    """

    b = int(frames)
    warm = int(warmup_symbols)
    payload = int(payload_symbols)
    if b <= 0 or warm < 1 or payload < 2:
        raise ValueError("invalid trace dimensions")
    total = warm + payload
    rs = float(symbol_rate_baud)
    linewidth = float(combined_linewidth_hz)
    if rs <= 0 or linewidth < 0:
        raise ValueError("invalid physical parameters")

    # Sign-labelled QPSK.  Fourth-power processing removes the data exactly in
    # the noiseless case, so these random labels are only needed to preserve the
    # correct AWGN/CPE interaction.
    bits = rng.integers(0, 2, size=(b, 2 * total), dtype=np.uint8)
    i = 1.0 - 2.0 * bits[:, 0::2].astype(float)
    q = 1.0 - 2.0 * bits[:, 1::2].astype(float)
    x = (i + 1j * q) / np.sqrt(2.0)

    phase_std = float(np.sqrt(2.0 * np.pi * linewidth / rs))
    theta = np.cumsum(rng.normal(0.0, phase_std, size=(b, total)), axis=1)
    n0 = float(10.0 ** (-float(snr_db) / 10.0))
    noise = np.sqrt(n0 / 2.0) * (
        rng.standard_normal((b, total)) + 1j * rng.standard_normal((b, total))
    )
    y = x * np.exp(1j * theta) + noise

    unit = y / np.maximum(np.abs(y), np.finfo(float).tiny)
    fourth = -(unit ** 4)  # sign-labelled QPSK has x^4=-1
    avg = _moving_average_causal(fourth, vv_window)
    principal = np.angle(avg) / 4.0

    # Nearest-branch continuation.  Vectorized over frames; the symbol loop is
    # short and deterministic.
    est = np.empty_like(principal)
    est[:, 0] = principal[:, 0]
    step = np.pi / 2.0
    for t in range(1, total):
        k = np.rint((est[:, t - 1] - principal[:, t]) / step)
        est[:, t] = principal[:, t] + k * step

    # Quadrant offset relative to the true laser phase.  This is simulation-only
    # metadata used to classify whether the physical CPE trace contains a slip.
    quadrant = np.rint((est - theta) / step).astype(np.int16)
    q0 = quadrant[:, warm][:, None]
    est_anchored = est - q0 * step
    corrected = y * np.exp(-1j * est_anchored)
    gain = corrected / x
    gain_payload = gain[:, warm : warm + payload]

    qrel = quadrant[:, warm : warm + payload] - quadrant[:, warm][:, None]
    dq = np.diff(qrel, axis=1)
    change_count = np.count_nonzero(dq, axis=1)
    final = qrel[:, -1]
    one = (change_count == 1) & (np.abs(final) == 1)
    no = change_count == 0
    event_class = np.full(b, 2, dtype=np.int8)
    event_class[no] = 0
    event_class[one] = 1

    locations = np.full(b, -1, dtype=np.int16)
    directions = np.zeros(b, dtype=np.int8)
    one_idx = np.flatnonzero(one)
    if one_idx.size:
        first_change = np.argmax(dq[one_idx] != 0, axis=1) + 1
        locations[one_idx] = first_change.astype(np.int16)
        # Corrected residual is approximately -qrel*pi/2.
        directions[one_idx] = ((-final[one_idx]) % 4).astype(np.int8)

    conf = np.abs(avg[:, warm : warm + payload])
    est_payload = est_anchored[:, warm : warm + payload]
    return VVTraceBatch(
        gain=gain_payload.astype(np.complex128, copy=False),
        event_class=event_class,
        slip_location=locations,
        slip_direction=directions,
        cpe_confidence_min=np.min(conf, axis=1),
        cpe_confidence_mean=np.mean(conf, axis=1),
        cpe_estimate_range=np.ptp(est_payload, axis=1),
    )


def collect_trace_pool(
    rng: np.random.Generator,
    *,
    no_slip_target: int,
    one_slip_target: int,
    other_target: int,
    batch_frames: int,
    max_total_frames: int,
    warmup_symbols: int,
    payload_symbols: int,
    symbol_rate_baud: float,
    combined_linewidth_hz: float,
    snr_db: float,
    vv_window: int,
) -> TracePool:
    no_parts: list[np.ndarray] = []
    one_parts: list[np.ndarray] = []
    other_parts: list[np.ndarray] = []
    loc_parts: list[np.ndarray] = []
    dir_parts: list[np.ndarray] = []
    no_count = one_count = other_count = total = 0
    observed_no = observed_one = observed_other = 0
    conf_no: list[np.ndarray] = []
    conf_one: list[np.ndarray] = []

    while total < int(max_total_frames) and (
        no_count < int(no_slip_target)
        or one_count < int(one_slip_target)
        or other_count < int(other_target)
    ):
        current = min(int(batch_frames), int(max_total_frames) - total)
        batch = generate_vv_trace_batch(
            rng,
            frames=current,
            warmup_symbols=warmup_symbols,
            payload_symbols=payload_symbols,
            symbol_rate_baud=symbol_rate_baud,
            combined_linewidth_hz=combined_linewidth_hz,
            snr_db=snr_db,
            vv_window=vv_window,
        )
        total += current
        ino = np.flatnonzero(batch.event_class == 0)
        ione = np.flatnonzero(batch.event_class == 1)
        iother = np.flatnonzero(batch.event_class == 2)
        observed_no += int(ino.size)
        observed_one += int(ione.size)
        observed_other += int(iother.size)

        if no_count < no_slip_target and ino.size:
            take = ino[: max(0, no_slip_target - no_count)]
            no_parts.append(batch.gain[take])
            conf_no.append(batch.cpe_confidence_min[take])
            no_count += take.size
        if one_count < one_slip_target and ione.size:
            take = ione[: max(0, one_slip_target - one_count)]
            one_parts.append(batch.gain[take])
            loc_parts.append(batch.slip_location[take])
            dir_parts.append(batch.slip_direction[take])
            conf_one.append(batch.cpe_confidence_min[take])
            one_count += take.size
        if other_count < other_target and iother.size:
            take = iother[: max(0, other_target - other_count)]
            other_parts.append(batch.gain[take])
            other_count += take.size

    def stack(parts: list[np.ndarray], shape_tail: tuple[int, ...], dtype: Any) -> np.ndarray:
        if not parts:
            return np.empty((0, *shape_tail), dtype=dtype)
        return np.concatenate(parts, axis=0)

    no_gain = stack(no_parts, (payload_symbols,), np.complex128)
    one_gain = stack(one_parts, (payload_symbols,), np.complex128)
    other_gain = stack(other_parts, (payload_symbols,), np.complex128)
    loc = np.concatenate(loc_parts) if loc_parts else np.empty(0, dtype=np.int16)
    direction = np.concatenate(dir_parts) if dir_parts else np.empty(0, dtype=np.int8)

    summary = {
        "total_frames_generated": int(total),
        "no_slip_collected": int(no_gain.shape[0]),
        "one_slip_collected": int(one_gain.shape[0]),
        "other_collected": int(other_gain.shape[0]),
        "observed_no_slip_frames": int(observed_no),
        "observed_one_slip_frames": int(observed_one),
        "observed_other_frames": int(observed_other),
        "one_slip_rate_point_estimate": float(observed_one / max(total, 1)),
        "symbol_rate_baud": float(symbol_rate_baud),
        "combined_linewidth_hz": float(combined_linewidth_hz),
        "normalized_linewidth": float(combined_linewidth_hz / symbol_rate_baud),
        "phase_innovation_std_rad": float(np.sqrt(2.0 * np.pi * combined_linewidth_hz / symbol_rate_baud)),
        "snr_db": float(snr_db),
        "vv_window": int(vv_window),
        "warmup_symbols": int(warmup_symbols),
        "payload_symbols": int(payload_symbols),
        "median_min_cpe_confidence_no_slip": float(np.median(np.concatenate(conf_no))) if conf_no else float("nan"),
        "median_min_cpe_confidence_one_slip": float(np.median(np.concatenate(conf_one))) if conf_one else float("nan"),
    }
    return TracePool(
        gain_no_slip=no_gain,
        gain_one_slip=one_gain,
        gain_other=other_gain,
        one_slip_locations=loc,
        one_slip_directions=direction,
        summary=summary,
    )
