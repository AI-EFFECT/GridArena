"""Leakage-safe train / validation / test split suggestions (Phase 2, part 2d).

A random split of a run's samples leaks: every topology variant of a
timestamp shares its load, and neighbouring hours look alike, so a model can
score well on the test set by remembering the training set. These splits keep
related samples together:

- `time`: whole calendar days of the historical timestamps -- every variant
  and every hour of a day in the same split;
- `topology`: whole configurations when the run reconfigured, otherwise whole
  topology variants (outage sets). The base topology always goes to training,
  so the test topologies are ones the model never saw;
- groups (several grids): whole grids, stratified by size (node count).

Units are shuffled with the seed and filled into the splits in turn until
each holds its share of the samples, so the split sizes follow the ratios in
samples, not in units.
"""

from typing import Dict, Hashable, List, Optional, Sequence, Tuple

import numpy as np

SPLITS = ("train", "val", "test")


def _ratios(train: float, val: float) -> Tuple[float, float, float]:
    if not (0 < train < 1 and 0 <= val < 1 and train + val <= 1):
        raise ValueError("Split ratios need 0 < train < 1, 0 <= val and train + val <= 1.")
    return train, val, 1 - train - val


def assign_units(sizes: Dict[Hashable, int], train: float, val: float, seed: int,
                 forced_train: Sequence[Hashable] = ()) -> Dict[Hashable, str]:
    """Split label per unit: units (forced ones first) fill train, then val,
    then test, each until it holds its share of the total size."""
    ratios = _ratios(train, val)
    total = sum(sizes.values())
    order = [u for u in forced_train if u in sizes]
    rest = [u for u in sizes if u not in set(order)]
    rng = np.random.default_rng(seed)
    order += [rest[i] for i in rng.permutation(len(rest))]
    bounds = np.cumsum(ratios) * total
    out, filled = {}, 0
    for u in order:
        # The split whose share the unit's midpoint falls in.
        mid = filled + sizes[u] / 2
        k = int(np.searchsorted(bounds, mid, side="right"))
        out[u] = SPLITS[min(k, 2)] if u not in forced_train else "train"
        filled += sizes[u]
    # With few units the shares can leave a split empty: give each split with a
    # positive ratio one unit, taken from the split with the most units.
    for k, sp in enumerate(SPLITS):
        if ratios[k] <= 0 or sp in out.values():
            continue
        held = {d: [u for u in order if out[u] == d] for d in SPLITS}
        movable = {d: [u for u in held[d] if u not in forced_train] for d in SPLITS}
        donors = [d for d in SPLITS if len(held[d]) > 1 and movable[d]]
        if donors:
            donor = max(donors, key=lambda d: len(held[d]))
            out[movable[donor][-1]] = sp
    return out


def split_samples(rows: Sequence[tuple], by: str, train: float = 0.7, val: float = 0.15,
                  seed: int = 0) -> dict:
    """rows: (scenario, load_scenario_idx, topology_variant_idx, scenario_timestamp,
    config_idx, kind) per runtime row (database.get_pf_datagen_split_rows)."""
    if not rows:
        raise ValueError("The run has no samples to split.")
    if by == "time":
        unit_of = [str(r[3])[:10] for r in rows]                    # calendar day
        forced: List = []
        unit_name = "day"
    elif by == "topology":
        reconfigured = any((r[4] or 0) > 0 for r in rows)
        if reconfigured:
            unit_of = [r[4] or 0 for r in rows]
            forced = [0]                                              # the base configuration
        else:
            unit_of = [r[2] for r in rows]
            forced = sorted({r[2] for r in rows if r[5] == "base"})
        unit_name = "configuration" if reconfigured else "topology_variant"
    else:
        raise ValueError("by must be 'time' or 'topology'.")
    sizes: Dict = {}
    for u in unit_of:
        sizes[u] = sizes.get(u, 0) + 1
    label = assign_units(sizes, train, val, seed, forced)
    samples = [{"scenario": r[0], "load_scenario_idx": r[1], "topology_variant_idx": r[2],
                "scenario_timestamp": r[3], "split": label[u]} for r, u in zip(rows, unit_of)]
    counts = {s: 0 for s in SPLITS}
    units = {s: [] for s in SPLITS}
    for s_ in samples:
        counts[s_["split"]] += 1
    for u, sp in label.items():
        units[sp].append(u)
    note = None
    if len(sizes) < 3:
        note = f"Only {len(sizes)} {unit_name}(s): not every split can get one."
    return {"by": by, "unit": unit_name, "ratios": dict(zip(SPLITS, _ratios(train, val))), "seed": seed,
            "counts": counts, "units": {k: sorted(v, key=str) for k, v in units.items()}, "note": note,
            "samples": samples}


def split_grids(sizes: Dict[str, int], train: float = 0.7, val: float = 0.15, seed: int = 0,
                weights: Optional[Dict[str, int]] = None) -> dict:
    """Grids to splits, stratified by size: grids sorted by node count are cut
    into blocks of about 1 / min(val, test) grids, and each block is split by
    the ratios (largest remainder), so every split gets small and large grids.
    `weights` (samples per grid) only feed the reported counts."""
    ratios = _ratios(train, val)
    grids = sorted(sizes, key=lambda g: (sizes[g], g))
    smallest = min(r for r in ratios if r > 0) if any(r > 0 for r in ratios) else 1
    block = max(len(SPLITS), int(round(1 / smallest)))
    rng = np.random.default_rng(seed)
    out: Dict[str, str] = {}
    carry = np.zeros(3)
    for start in range(0, len(grids), block):
        chunk = grids[start:start + block]
        exact = np.array(ratios) * len(chunk) + carry
        n = np.floor(exact).astype(int)
        for k in np.argsort(-(exact - n), kind="stable")[: len(chunk) - n.sum()]:
            n[k] += 1
        carry = exact - n
        labels = [SPLITS[k] for k in range(3) for _ in range(n[k])]
        for g, i in zip(chunk, rng.permutation(len(chunk))):
            out[g] = labels[i]
    counts = {s: {"grids": 0, "samples": 0} for s in SPLITS}
    for g, sp in out.items():
        counts[sp]["grids"] += 1
        counts[sp]["samples"] += (weights or {}).get(g, 0)
    return {"by": "grid", "ratios": dict(zip(SPLITS, ratios)), "seed": seed, "grids": out, "counts": counts,
            "node_counts": {g: sizes[g] for g in grids}}
