"""Joint sampling design over the continuous axes (Phase 2, part 2d).

Each axis already draws its per-sample values stratified, in shuffled order,
from its own stream (`sampling.draw_values`): per axis that is a Latin
hypercube, but the pairing between axes is left to chance, so a short run can
pair high load with high penetration more often than it should, or leave some
combinations empty. `design.mode` controls the pairing:

- `independent` (default): each axis as before;
- `lhs`: one Latin hypercube over the axes in the design -- n strata per axis
  -- keeping, of LHS_TRIES random pairings, the one with the smallest largest
  rank correlation between two axes;
- `factorial`: every combination of the axes' bins filled equally (cell
  counts differ by at most one); needs at least as many samples as cells.

The axes in the design are the load level (when on) and the busbar
set-point (in `uniform` mode, one value for all substations); generated
DER have a fixed capacity (der.groups), so no penetration axis. Values are drawn on [0, 1) and mapped to each
axis's range; the draws come from stream 15.

`cells` labels every sample with its stratum -- the axes' own bins -- which
the target mix (target_mix.py) and the report use, whatever the mode.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from gridarena.pf_datagen.sampling import STREAM_DESIGN, bin_of, stream

LHS_TRIES = 50


@dataclass
class Axis:
    name: str              # "level" | "penetration" | "slack"
    lo: float
    hi: float
    bins: int


@dataclass
class Design:
    mode: str
    axes: List[Axis]
    values: Dict[str, np.ndarray] = field(default_factory=dict)   # axis name -> per-sample value

    def cells(self, values: Optional[Dict[str, np.ndarray]] = None) -> List[Optional[str]]:
        """Per-sample stratum label, e.g. "L3.P7" (bins of each axis, in order)."""
        values = values or self.values
        if not self.axes:
            return [None] * self._n(values)
        idx = [bin_of(values[a.name], a.lo, a.hi, a.bins) for a in self.axes]
        tags = [a.name[0].upper() for a in self.axes]
        return [".".join(f"{t}{int(b[i])}" for t, b in zip(tags, idx)) for i in range(len(idx[0]))]

    @staticmethod
    def _n(values):
        return len(next(iter(values.values()))) if values else 0

    def max_rank_correlation(self) -> Optional[float]:
        """Largest |Spearman correlation| between two axes of the design."""
        if len(self.axes) < 2:
            return None
        ranks = [np.argsort(np.argsort(self.values[a.name])) for a in self.axes]
        best = 0.0
        for i in range(len(ranks)):
            for j in range(i + 1, len(ranks)):
                if np.std(ranks[i]) > 0 and np.std(ranks[j]) > 0:
                    best = max(best, abs(float(np.corrcoef(ranks[i], ranks[j])[0, 1])))
        return best


def lhs_unit(n: int, d: int, rng: np.random.Generator, tries: int = LHS_TRIES) -> np.ndarray:
    """n points in [0, 1)^d, one per stratum of 1/n on every axis, pairing chosen
    to keep the largest rank correlation between two axes small."""
    if d == 0 or n == 0:
        return np.zeros((n, d))
    best, best_score = None, np.inf
    base = np.arange(n)
    for _ in range(max(1, tries)):
        perms = np.column_stack([base] + [rng.permutation(n) for _ in range(d - 1)])
        if d < 2 or n < 3:
            best = perms
            break
        c = np.corrcoef(perms, rowvar=False)
        score = np.max(np.abs(c[np.triu_indices(d, 1)]))
        if score < best_score:
            best, best_score = perms, score
    u = (best + rng.uniform(0.0, 1.0, size=(n, d))) / n
    return u[rng.permutation(n)]             # rows in shuffled order, not following the timeline


def factorial_unit(n: int, bins: Sequence[int], rng: np.random.Generator) -> np.ndarray:
    """n points in [0, 1)^d filling every cell of the bins' grid equally."""
    d = len(bins)
    n_cells = int(np.prod(bins)) if d else 1
    if d == 0:
        return np.zeros((n, 0))
    if n < n_cells:
        raise ValueError(f"design.mode 'factorial' needs at least as many samples as cells: {n} samples for "
                         f"{n_cells} cells ({' x '.join(str(b) for b in bins)} bins). Raise scenario_count, "
                         "lower the bins, or use 'lhs'.")
    cell = np.concatenate([np.repeat(np.arange(n_cells), n // n_cells),
                           rng.choice(n_cells, size=n % n_cells, replace=False)])
    cell = rng.permutation(cell)
    idx = np.array(np.unravel_index(cell, bins)).T             # (n, d) bin index per axis
    return (idx + rng.uniform(0.0, 1.0, size=(n, d))) / np.asarray(bins, dtype=float)


def build_design(mode: str, axes: List[Axis], n: int, seed: int) -> Optional[Design]:
    """The joint draws for `axes`, or None in `independent` mode (each axis
    then draws its own values)."""
    if mode == "independent" or not axes:
        return None
    rng = stream(seed, STREAM_DESIGN)
    if mode == "lhs":
        u = lhs_unit(n, len(axes), rng)
    elif mode == "factorial":
        u = factorial_unit(n, [a.bins for a in axes], rng)
    else:
        raise ValueError(f"Unknown design.mode '{mode}'.")
    values = {a.name: a.lo + u[:, k] * (a.hi - a.lo) for k, a in enumerate(axes)}
    return Design(mode, axes, values)


def variant_order(design: Optional[Design], n: int) -> Optional[np.ndarray]:
    """Position of each sample when spreading samples over variants
    (outages.assign_variants): load-level rank (else penetration rank) under a
    joint design, so every variant sees the whole range; None otherwise."""
    if design is None:
        return None
    for name in ("level", "penetration", "slack"):
        if name in design.values:
            return np.argsort(np.argsort(design.values[name], kind="stable"), kind="stable")
    return None
