"""Target mix (Phase 2, part 2d): a minimum share of samples with a violation.

Violations are rare in typical operation, so a dataset drawn evenly over the
load range may hold few of them. With `target_mix.min_violation_share` set, a
run first solves a pilot -- a random `pilot_share` of its samples -- and
measures the violation rate r_c of each stratum c (the load-level bins, times
the penetration bins when DER are generated). The remaining samples are then
drawn in ROUNDS rounds over the strata with weights

    w = FLOOR * uniform + (1 - FLOOR) * q,   q_c proportional to r_c ** k,

taking the smallest tilt k in [0, MAX_TILT] that brings the run's share of
violating samples to the target (k = MAX_TILT when even that falls short; the
report says so). k = 0 is the even design; larger k leans towards the strata
that violate most. The floor keeps every stratum at least FLOOR of its even
share, so the data still covers them and no design_weight exceeds 1 / FLOOR.
Before each round the rates are re-estimated from every sample solved so far,
each shrunk towards the overall rate as if SHRINK more samples at that rate
had been seen: a pilot of a few samples per stratum is too noisy to steer by
alone. Within a stratum, values stay uniform.

Oversampling changes the distribution, so every sample carries a
`design_weight`: the uniform share of its stratum over the run's actual share
(pilot and rest together). Weighted statistics then estimate what an even
design would give.

A sample violates when it converged with an under- or overvoltage
(`violations: voltage`), a thermal violation (`thermal`) or either (`any`).
Strata the pilot didn't visit take the pilot's overall rate.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from gridarena.pf_datagen.sampling import STREAM_TARGET_MIX, sample_stream

ROUNDS = 3
SHRINK = 2.0
FLOOR = 0.2
MAX_TILT = 8.0


def pilot_ids(n: int, share: float, seed: int) -> List[int]:
    """The pilot's samples: a random subset of round(share * n) (at least 1)."""
    k = min(n, max(1, int(round(share * n))))
    return sorted(int(i) for i in sample_stream(seed, STREAM_TARGET_MIX, 0).choice(n, size=k, replace=False))


def rounds(rest: Sequence[int], seed: int, n_rounds: int = ROUNDS) -> List[List[int]]:
    """The samples after the pilot, in random order, cut into n_rounds batches."""
    order = [rest[i] for i in sample_stream(seed, STREAM_TARGET_MIX, 2).permutation(len(rest))]
    return [sorted(b) for b in np.array_split(np.array(order, dtype=int), n_rounds) if len(b)]


def is_violation(summary_row: Optional[tuple], kind: str, s_index: Dict[str, int]) -> bool:
    if summary_row is None:
        return False
    volt = (summary_row[s_index["n_undervoltage"]] or 0) + (summary_row[s_index["n_overvoltage"]] or 0) > 0
    thermal = (summary_row[s_index["n_thermal_violations"]] or 0) > 0
    return {"voltage": volt, "thermal": thermal}.get(kind, volt or thermal)


@dataclass
class Retarget:
    """The weights chosen after the pilot, and how far they reach."""
    strata: List[str]
    pilot_rate: Dict[str, Optional[float]]
    pilot_outputs: Dict[str, int]
    tilt: float
    weights: Dict[str, float]
    expected_share: float
    reachable: bool
    counts: Dict[str, int] = field(default_factory=dict)        # remaining samples per stratum


def choose_weights(strata: Sequence[str], outcomes: Sequence[tuple], target: float, n_pilot: int,
                   n_rest: int, shrink: float = SHRINK) -> Retarget:
    """outcomes: (stratum, violated or None when not converged) per output
    solved so far (n_pilot samples); n_rest samples are still to draw."""
    strata = list(strata)
    per = {c: [0, 0] for c in strata}                    # violating, converged outputs
    for c, v in outcomes:
        if v is None or c not in per:
            continue
        per[c][1] += 1
        per[c][0] += bool(v)
    total_v = sum(p[0] for p in per.values())
    total_c = sum(p[1] for p in per.values())
    overall = total_v / total_c if total_c else 0.0
    rate = {c: (per[c][0] + shrink * overall) / (per[c][1] + shrink) for c in strata}
    r = np.array([rate[c] for c in strata])
    u = np.full(len(strata), 1.0 / len(strata))

    def weights(k: float) -> np.ndarray:
        if r.max() <= 0:
            return u
        q = (r / r.max()) ** k
        return FLOOR * u + (1 - FLOOR) * q / q.sum()

    # Share needed among the remaining samples for the whole run to reach the target.
    share_so_far = total_v / total_c if total_c else 0.0
    n_all = n_pilot + n_rest
    needed = (target * n_all - share_so_far * n_pilot) / n_rest if n_rest else 0.0
    if float(weights(0.0) @ r) >= needed:
        tilt = 0.0
    elif float(weights(MAX_TILT) @ r) < needed:
        tilt = MAX_TILT
    else:
        lo, hi = 0.0, MAX_TILT
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if float(weights(mid) @ r) < needed else (lo, mid)
        tilt = hi
    w = weights(tilt)
    share_rest = float(w @ r)
    expected = (share_so_far * n_pilot + share_rest * n_rest) / n_all if n_all else 0.0
    return Retarget(strata, {c: (per[c][0] / per[c][1] if per[c][1] else None) for c in strata},
                    {c: per[c][1] for c in strata}, tilt, dict(zip(strata, w.tolist())), expected,
                    share_rest >= needed - 1e-9)


def allocate(weights: Dict[str, float], n: int, rng: np.random.Generator) -> List[str]:
    """n stratum labels in proportion to `weights` (largest remainder), shuffled."""
    keys = list(weights)
    w = np.array([weights[k] for k in keys])
    exact = w / w.sum() * n
    counts = np.floor(exact).astype(int)
    for k in np.argsort(-(exact - counts), kind="stable")[: n - counts.sum()]:
        counts[k] += 1
    labels = [k for k, c in zip(keys, counts) for _ in range(c)]
    return [labels[i] for i in rng.permutation(len(labels))]


def design_weights(cells: Sequence[Optional[str]], strata: Sequence[str]) -> List[Optional[float]]:
    """Uniform share of each sample's stratum over its share in the run."""
    n = len(cells)
    counts: Dict[str, int] = {}
    for c in cells:
        counts[c] = counts.get(c, 0) + 1
    uniform = 1.0 / len(strata)
    return [uniform / (counts[c] / n) if c in counts and c is not None else None for c in cells]
