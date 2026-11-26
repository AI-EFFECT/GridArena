"""Synthetic MV feeder generator (Phase 2, axis 7B).

A grid is grown node by node from its substation(s), calibrated per archetype
(calibrate.py):

- **Shape.** Each feeder starts as a trunk from the substation. Every new node
  starts a lateral with the lateral probability, off a node that continues
  onwards but doesn't branch yet (so every lateral adds a branch point, as on
  real feeders, instead of lengthening a spur or piling onto one junction); otherwise, while a lateral is being built, it goes
  back to extending the trunk with the return probability, and else extends
  the branch being built. A lateral started off a lateral is a sub-lateral.
  High return gives a long trunk with short spurs (rural feeders), low return
  long laterals. Both probabilities are calibrated (calibrate.py) to the
  archetype's share of branching nodes and its depth relative to size. Nodes
  are split evenly over the feeders.
- **Geometry.** Each section keeps its parent's heading within a few degrees;
  a lateral leaves at 60-90 degrees. Section lengths come from the
  archetype's log-normal fit; the cable is ROUTING_FACTOR times the
  straight-line distance. Positions are placed around the reference location.
- **Equipment.** Every branch end is a connection point; other nodes are
  connection points with the probability that gives the archetype's share.
  Each connection point gets one MV/LV transformer, its rating drawn from the
  archetype's mix. Whole branches are made overhead, in random order, until
  their length reaches the grid's overhead share (drawn from the archetype's
  grids); every section starts on its family's smaller cable.
- **Beyond the stored grids** (options): 1-4 feeders per substation, a second
  substation joined by a normally-open tie, and 0-3 normally-open ties between
  feeders -- each between the closest pair of nodes on different feeders.
  Each feeder's first section and the section into each tie end are closed
  switches (batch.py), so a tie's loop can be reconfigured.

The planning check (planning.py) then upgrades cables until the grid carries
its design load, or the grid is drawn again.
"""

from dataclasses import dataclass, field
from math import cos, radians, sin
from typing import Dict, List, Optional, Tuple

import numpy as np

GENERATOR_VERSION = 1
MAX_NODES = 500
ROUTING_FACTOR = 1.2
REFERENCE = (48.85, 2.2)
M_PER_DEG = 111195.0
HEADING_SIGMA_DEG = 15.0
SECTION_RANGE_M = (20.0, 3000.0)
NOMINAL_KV = 20.0


@dataclass
class Options:
    feeders_per_substation: int = 1
    second_substation: bool = False
    ties_between_feeders: int = 0
    n_nodes: Optional[Tuple[int, int]] = None      # fixes the node count range instead of the fit
    reference: Tuple[float, float] = REFERENCE


@dataclass
class Skeleton:
    """A generated grid before cables are sized: node 0.. with parent links."""
    parent: List[int]                      # -1 for substations
    feeder: List[int]                      # feeder index (-1 for substations)
    branch: List[int]                      # branch index the node's section belongs to
    xy: np.ndarray                         # (n, 2) metres from the reference
    length: List[float]                    # cable length of the section to the parent (0 for substations)
    substations: List[int]
    ties: List[Tuple[int, int, float]] = field(default_factory=list)    # (a, b, cable length)


def _grow_feeder(n: int, p_lateral: float, p_return: float,
                 rng: np.random.Generator) -> Tuple[List[int], List[int]]:
    """Parent (local index, -1 = substation) and branch id of n feeder nodes;
    branch 0 is the trunk."""
    parent, branch = [], []
    tip, trunk_tip, current, n_branches = -1, -1, 0, 0
    inner: List[int] = []                         # nodes with exactly one child: a lateral off one adds a branch point
    children = [0] * n
    for k in range(n):
        if k > 0 and inner and rng.random() < p_lateral:
            tip = inner.pop(int(rng.integers(0, len(inner))))
            n_branches += 1
            current = n_branches
        elif current != 0 and rng.random() < p_return:
            tip, current = trunk_tip, 0
        if tip >= 0:
            children[tip] += 1
            if children[tip] == 1:
                inner.append(tip)                 # its first child: it now continues onwards
        parent.append(tip)
        branch.append(current)
        tip = k
        if current == 0:
            trunk_tip = k
    return parent, branch


def branching_share(parents: List[int], n_total: int) -> float:
    deg = np.zeros(n_total + 1, dtype=int)
    for k, p in enumerate(parents):
        deg[k + 1] += 1
        deg[p + 1] += 1
    return float((deg[1:] >= 3).sum()) / (n_total + 1)


def _depth(parents: List[int]) -> int:
    depth = []
    for p in parents:
        depth.append(1 if p < 0 else depth[p] + 1)
    return max(depth, default=0)


def calibrate_growth(fit, seed: int = 0) -> Tuple[float, float]:
    """The (lateral, return) probabilities whose medians of branching share
    and depth / node count, over feeders of the archetype's sizes, come
    closest to its stored grids (relative errors, summed)."""
    rng = np.random.default_rng([seed, 7])
    sizes = np.clip(np.exp(rng.normal(*fit.log_nodes, size=30)), 5, MAX_NODES).astype(int)
    best, best_err = (0.0, 0.0), np.inf
    for p in np.linspace(0.0, 0.9, 31):
        for r in np.linspace(0.0, 1.0, 11):
            shares, ratios = [], []
            for k, n in enumerate(sizes):
                parents, _ = _grow_feeder(int(n) - 1, p, r, np.random.default_rng([seed, 8, k]))
                shares.append(branching_share(parents, int(n) - 1))
                ratios.append(_depth(parents) / n)
            err = (abs(float(np.median(shares)) - fit.branching) / max(fit.branching, 0.05)
                   + abs(float(np.median(ratios)) - fit.depth_ratio) / max(fit.depth_ratio, 0.05))
            if err < best_err:
                best, best_err = (float(p), float(r)), err
    return best


def _lognormal(rng, params, lo, hi, size=None):
    return np.clip(np.exp(rng.normal(params[0], params[1], size=size)), lo, hi)


def draw_size(fit, opts: Options, rng: np.random.Generator) -> Tuple[int, float]:
    """A grid's node count and typical (median) section length. Drawn once per
    grid: a grid that fails the planning check is grown again at the same
    size, or larger grids, which fail more often, would end up too rare."""
    if opts.n_nodes is not None:
        n_total = int(rng.integers(opts.n_nodes[0], opts.n_nodes[1] + 1))
    else:
        n_total = int(_lognormal(rng, fit.log_nodes, 5, MAX_NODES))
    return n_total, float(_lognormal(rng, fit.log_section, *SECTION_RANGE_M))


def grow(fit, opts: Options, rng: np.random.Generator, size: Optional[Tuple[int, float]] = None) -> Skeleton:
    n_total, median_m = size if size is not None else draw_size(fit, opts, rng)
    n_subs = 2 if opts.second_substation else 1
    n_feeders = n_subs * opts.feeders_per_substation
    n_feeder_nodes = max(n_feeders, n_total - n_subs)
    split = np.full(n_feeders, n_feeder_nodes // n_feeders)
    split[: n_feeder_nodes % n_feeders] += 1

    parent, feeder, branch, length = [-1] * n_subs, [-1] * n_subs, [-1] * n_subs, [0.0] * n_subs
    xy = [np.zeros(2)]
    if n_subs == 2:
        # The second substation a typical feeder reach away.
        d = median_m * 25
        a = rng.uniform(0, 2 * np.pi)
        xy.append(np.array([d * cos(a), d * sin(a)]))
    heading = [0.0] * n_subs
    branch_offset = 0
    for f in range(n_feeders):
        sub = f // opts.feeders_per_substation
        base = 2 * np.pi * (f % opts.feeders_per_substation) / opts.feeders_per_substation + rng.uniform(-0.3, 0.3)
        local_parent, local_branch = _grow_feeder(int(split[f]), fit.p_lateral, fit.p_return, rng)
        first = len(parent)
        lengths = np.clip(median_m * np.exp(rng.normal(0.0, fit.section_spread, size=len(local_parent))),
                          *SECTION_RANGE_M)
        for k, (lp, lb) in enumerate(zip(local_parent, local_branch)):
            p = sub if lp < 0 else first + lp
            if lp < 0:
                h = base
            elif local_branch[k] != local_branch[lp]:
                h = heading[p] + rng.choice([-1, 1]) * radians(rng.uniform(60, 90))      # a new lateral
            else:
                h = heading[p] + radians(rng.normal(0, HEADING_SIGMA_DEG))
            step = lengths[k] / ROUTING_FACTOR
            xy.append(xy[p] + step * np.array([cos(h), sin(h)]))
            heading.append(h)
            parent.append(p)
            feeder.append(f)
            branch.append(branch_offset + lb)
            length.append(float(lengths[k]))
        branch_offset += max(local_branch) + 1
    sk = Skeleton(parent, feeder, branch, np.array(xy), length, list(range(n_subs)))
    _add_ties(sk, opts)
    return sk


def _add_ties(sk: Skeleton, opts: Options) -> None:
    """Normally-open ties between the closest nodes of different feeders: one
    joining the two substations' feeders (with a second substation), then
    `ties_between_feeders` more, no node used twice."""
    nodes = [i for i in range(len(sk.parent)) if sk.feeder[i] >= 0]
    if len(set(sk.feeder[i] for i in nodes)) < 2:
        return
    pts = sk.xy[nodes]
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    fe = np.array([sk.feeder[i] for i in nodes])
    sub_of = fe // opts.feeders_per_substation
    used = set()

    def best(mask):
        dm = np.where(mask & (fe[:, None] != fe[None, :]), d, np.inf)
        for i in list(used):
            k = nodes.index(i)
            dm[k, :] = np.inf
            dm[:, k] = np.inf
        k = int(np.argmin(dm))
        a, b = divmod(k, len(nodes))
        if not np.isfinite(dm[a, b]):
            return False
        sk.ties.append((nodes[a], nodes[b], max(float(dm[a, b]) * ROUTING_FACTOR, 20.0)))
        used.update((nodes[a], nodes[b]))
        return True

    if opts.second_substation:
        best(sub_of[:, None] != sub_of[None, :])
    for _ in range(opts.ties_between_feeders):
        if not best(np.ones_like(d, dtype=bool)):
            break


def to_lat_lon(xy: np.ndarray, reference: Tuple[float, float]) -> List[Tuple[float, float]]:
    lat0, lon0 = reference
    return [(lat0 + y / M_PER_DEG, lon0 + x / (M_PER_DEG * cos(radians(lat0)))) for x, y in xy]


def equip(sk: Skeleton, fit, rng: np.random.Generator) -> dict:
    """Connection points, transformer ratings and each section's family."""
    n = len(sk.parent)
    children = np.zeros(n, dtype=int)
    for p in sk.parent:
        if p >= 0:
            children[p] += 1
    feeder_nodes = [i for i in range(n) if sk.parent[i] >= 0]
    leaves = [i for i in feeder_nodes if children[i] == 0]
    others = [i for i in feeder_nodes if children[i] > 0]
    want = fit.cp_share * n
    q = float(np.clip((want - len(leaves)) / len(others), 0.0, 1.0)) if others else 0.0
    cps = sorted(set(leaves) | {i for i in others if rng.random() < q})
    kva_values = np.array(list(fit.ratings))
    kva_p = np.array([fit.ratings[k] for k in fit.ratings])
    ratings = {i: float(rng.choice(kva_values, p=kva_p / kva_p.sum())) for i in cps}
    oh_share = float(rng.choice(fit.overhead_shares))
    branch_len: Dict[int, float] = {}
    for i in feeder_nodes:
        branch_len[sk.branch[i]] = branch_len.get(sk.branch[i], 0.0) + sk.length[i]
    total = sum(branch_len.values())
    branch_oh, filled = {}, 0.0
    for b in rng.permutation(sorted(branch_len)):
        b = int(b)
        # Overhead while that brings the share closer to the target.
        take = abs(filled + branch_len[b] - oh_share * total) < abs(filled - oh_share * total)
        branch_oh[b] = bool(take)
        filled += branch_len[b] if take else 0.0
    family = {i: ("OH" if branch_oh[sk.branch[i]] else "UG") for i in feeder_nodes}
    return {"cps": cps, "ratings": ratings, "family": family, "overhead_share": oh_share}
