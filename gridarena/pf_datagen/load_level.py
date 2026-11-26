"""Load level per sample (Phase 2, axis 1).

Each sample's loads are scaled by one factor, P and Q together, so every node
keeps its share and power factor. Two ways to choose it:

- `multiplier`: a factor drawn from [min, max] on top of the historical load;
- `loadability`: a target total demand drawn as a fraction of P_max, the
  most demand the grid carries within its limits, so a level means the same
  thing on every grid (0.3 light, 0.9 near the limit, 1.1 beyond it).

P_max comes from a bisection on the run's average spatial load pattern: the
load is raised until the solve stops converging, the lowest voltage drops
below the run's minimum, or a branch exceeds its normal rating (`evaluate`,
supplied by the job, decides which). Demand is measured as gross demand, the
sum of positive P, so nodes exporting power don't cancel the load.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

# A sample whose gross demand is below this share of the run's median isn't
# rescaled (scaling a near-empty sample to a real level would blow up whatever
# little it has) and is flagged with an empty load_level.
NEAR_ZERO_SHARE = 0.01
SEARCH_REL_TOL = 0.01
MAX_BRACKET_STEPS = 40


@dataclass
class Loadability:
    p_max_w: float          # gross demand (W) at the limit
    scale: float            # P_max relative to the average pattern used for the search
    limit: str              # what stopped it: "voltage" | "thermal" | "convergence"
    n_solves: int


class LoadabilityError(Exception):
    """The grid is outside its limits even with (almost) no load."""


def gross_demand(s: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Sum of positive active power (W) in an injection vector (load
    convention), over the nodes in `mask` (all if None). The job passes the
    nodes the base topology energises, so load behind an out-of-service
    transformer, which is never served, doesn't count."""
    p = np.real(np.asarray(s))
    if mask is not None:
        p = p[mask]
    return float(p[p > 0].sum())


def find_loadability(pattern: np.ndarray, evaluate: Callable[[np.ndarray], Optional[str]],
                     rel_tol: float = SEARCH_REL_TOL, mask: Optional[np.ndarray] = None) -> Loadability:
    """Largest k such that `evaluate(pattern * k)` is within limits (returns
    None); `evaluate` returns the limit name otherwise. Brackets by doubling
    or halving from k = 1, then bisects to `rel_tol`."""
    g = gross_demand(pattern, mask)
    if g <= 0:
        raise LoadabilityError("The average load pattern has no demand; the load level can't be set.")
    n = 0

    def check(k):
        nonlocal n
        n += 1
        return evaluate(pattern * k)

    reason = check(1.0)
    if reason is None:
        lo, hi, hi_reason = 1.0, None, None
        k = 1.0
        for _ in range(MAX_BRACKET_STEPS):
            k *= 2.0
            r = check(k)
            if r is not None:
                hi, hi_reason = k, r
                break
            lo = k
        if hi is None:
            raise LoadabilityError("No limit found up to 2^40 times the average load; check the grid's ratings.")
    else:
        hi, hi_reason, lo = 1.0, reason, None
        k = 1.0
        for _ in range(MAX_BRACKET_STEPS):
            k /= 2.0
            r = check(k)
            if r is None:
                lo = k
                break
            hi, hi_reason = k, r
        if lo is None:
            raise LoadabilityError(
                f"The grid is outside its limits ({hi_reason}) even at a tiny fraction of its load: "
                "check the busbar set-point and the voltage limits."
            )
    while (hi - lo) / hi > rel_tol:
        mid = 0.5 * (lo + hi)
        r = check(mid)
        if r is None:
            lo = mid
        else:
            hi, hi_reason = mid, r
    return Loadability(p_max_w=lo * g, scale=lo, limit=hi_reason, n_solves=n)


def average_pattern(injections: List[np.ndarray]) -> np.ndarray:
    return np.mean(np.stack(injections), axis=0)


def apply_levels(injections: List[np.ndarray], mode: str, values: np.ndarray, p_max_w: Optional[float] = None,
                 mask: Optional[np.ndarray] = None) -> Tuple[List[np.ndarray], List[Optional[float]], List[float]]:
    """Scale each sample's injection vector. Returns (scaled vectors, level
    per sample, scale per sample). Vectors are new arrays.

    `multiplier`: scale = level = the drawn value. `loadability`: the sample
    is scaled so its gross demand is value x P_max; samples whose gross
    demand is near zero (NEAR_ZERO_SHARE of the median) stay as they are, with
    level None and scale 1."""
    if mode == "multiplier":
        scales = [float(v) for v in values]
        return [s * k for s, k in zip(injections, scales)], list(scales), scales
    gross = np.array([gross_demand(s, mask) for s in injections])
    positive = gross[gross > 0]
    floor = NEAR_ZERO_SHARE * float(np.median(positive)) if positive.size else np.inf
    out, levels, scales = [], [], []
    for s, g, v in zip(injections, gross, values):
        if g < floor or g <= 0:
            out.append(s.copy())
            levels.append(None)
            scales.append(1.0)
        else:
            k = float(v) * p_max_w / g
            out.append(s * k)
            levels.append(float(v))
            scales.append(k)
    return out, levels, scales
