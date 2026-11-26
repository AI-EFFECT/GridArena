"""Per-sample equipment outages (Phase 2 axis 4).

Unlike the topology variants (lines and transformers switched off, shared by
every sample solved on the variant), these are drawn for each sample i, and
shared by its variants:

- a DER unit trips (the grid's own and generated ones) with `der_trip_p`;
- a capacitor bank or reactor is out (no steps in) with `shunt_out_p`;
- a regulator is bypassed (held at tap 0, ratio 1) with `regulator_bypass_p`.

Each kind draws from its own stream [seed, 7, i, kind], so changing one
probability doesn't move the others' draws.
"""

from dataclasses import dataclass, replace
from typing import FrozenSet, List, Optional, Sequence

import numpy as np

from gridarena.pf_datagen.sampling import STREAM_OUTAGES, sample_stream


@dataclass(frozen=True)
class SampleOutage:
    ders_tripped: FrozenSet[str] = frozenset()
    shunts_out: FrozenSet[str] = frozenset()
    regulators_bypassed: FrozenSet[str] = frozenset()


def _draw(seed: int, i: int, kind: int, ids: Sequence[str], p: float) -> FrozenSet[str]:
    if p <= 0 or not ids:
        return frozenset()
    hit = sample_stream(seed, STREAM_OUTAGES, i, kind).random(len(ids)) < p
    return frozenset(x for x, h in zip(ids, hit) if h)


class OutageSampler:
    def __init__(self, cfg, seed: int):
        self.cfg, self.seed = cfg, seed

    @property
    def enabled(self) -> bool:
        c = self.cfg
        return c.der_trip_p > 0 or c.shunt_out_p > 0 or c.regulator_bypass_p > 0

    def sample(self, i: int, der_ids: Sequence[str], shunt_ids: Sequence[str],
               regulator_ids: Sequence[str]) -> SampleOutage:
        """The outages of sample i among the in-service units given (in a fixed order)."""
        c = self.cfg
        return SampleOutage(
            _draw(self.seed, i, 0, der_ids, c.der_trip_p),
            _draw(self.seed, i, 1, shunt_ids, c.shunt_out_p),
            _draw(self.seed, i, 2, regulator_ids, c.regulator_bypass_p),
        )


def bypass(regulators: List, bypassed: FrozenSet[str]) -> List:
    """Regulators with the bypassed ones pinned at tap 0 (ratio 1)."""
    if not bypassed:
        return regulators
    return [replace(r, tap_min=0, tap_max=0, tap_pos=0) if r.connection_id in bypassed else r
            for r in regulators]


def assign_variants(n_samples: int, n_variants: int, m: int, seed: int,
                    order: Optional[Sequence[int]] = None) -> List[List[int]]:
    """For variants_per_sample = m: the samples each variant is solved on.
    Each sample gets m distinct variants and every variant about the same
    number of samples (counts differ by at most one): sample i, at a random
    position in the run, takes m consecutive variants of a random
    permutation, wrapping round. With `order` (under a joint design, each
    sample's load-level rank) the position is that rank instead, so every
    variant's samples spread evenly over the load range."""
    from gridarena.pf_datagen.sampling import STREAM_VARIANT_CHOICE, stream

    m = min(m, n_variants)
    rng = stream(seed, STREAM_VARIANT_CHOICE)
    perm = rng.permutation(n_variants)
    position = rng.permutation(n_samples)
    if order is not None:
        position = np.asarray(order)
    out: List[List[int]] = [[] for _ in range(n_variants)]
    for i in range(n_samples):
        start = int(position[i]) * m
        for j in range(m):
            out[int(perm[(start + j) % n_variants])].append(i)
    return out
