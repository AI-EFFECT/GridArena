"""Topology and admittance perturbation for the MV PF data-generation pipeline.

MV feeders are operated radially: every closed connection of a feeder is a
cut-edge, so disabling one islands the part below it. (Normally-open
switches and tie lines are open in the base topology; closing them is
reconfiguration, proposal Phase 2b.) Rather than reject nearly every
candidate (the reference pipeline's "keep only if still a single connected
component, otherwise discard" rule, written for meshed transmission-style
test cases), a topology variant here is always accepted: the part still
connected to a substation is what gets solved, and the islanded nodes are
flagged in that variant's output. This mirrors what a real N-1 fault on a
radial feeder actually does.

Because no variant is ever rejected for islanding, `random` mode only redraws
to avoid duplicates: each variant disables a distinct set of connections.

With reconfiguration (Phase 2 axis 4, pf_datagen/reconfiguration.py) a variant
is a configuration plus an outage set: the outage sets are drawn for each
configuration over the connections closed in it. The base configuration's
outages come from the main generator, as before, so runs without
reconfiguration reproduce; configuration c >= 1 draws from [seed, 10, c].
"""

import itertools
from dataclasses import dataclass, field
from math import comb
from typing import AbstractSet, Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np

from gridarena.pf_datagen.grid_loader import MVBranch, substation_connected_component
from gridarena.pf_datagen.schemas import TopologyPerturbationConfig

MAX_N_MINUS_K_COMBOS = 5000


MAX_TOTAL_VARIANTS = 20000


@dataclass
class TopologyVariant:
    variant_idx: int
    disabled_connection_ids: Set[str] = field(default_factory=set)
    energised_node_ids: Set[str] = field(default_factory=set)
    config_idx: int = 0                                           # 0 = the base configuration
    opened_connection_ids: FrozenSet[str] = frozenset()           # closed in the base, opened by the configuration
    closed_connection_ids: FrozenSet[str] = frozenset()           # open in the base, closed by the configuration

    @property
    def kind(self) -> str:
        if self.config_idx == 0:
            return "outage" if self.disabled_connection_ids else "base"
        return "both" if self.disabled_connection_ids else "reconfiguration"

    def active_connection_ids(self, base_in_service: AbstractSet[str]) -> Set[str]:
        """Connections in service in this variant, from those in service in the base."""
        return ((set(base_in_service) - self.opened_connection_ids) | self.closed_connection_ids)             - self.disabled_connection_ids


def build_variants(
    config: TopologyPerturbationConfig, branches: List[MVBranch], node_ids: List[str],
    substation_node_ids: List[str], rng: np.random.Generator,
    candidates: Optional[List[MVBranch]] = None, out_of_service: AbstractSet[str] = frozenset(),
) -> List[TopologyVariant]:
    """Topology variants of the base grid. Outages are drawn from
    `candidates` (default: every branch); branches in `out_of_service` (e.g.
    an inactive transformer or a normally-open switch) are off in every
    variant without being listed as disabled, and energised nodes -- those
    connected to any substation -- are computed over all `branches`."""
    candidates = branches if candidates is None else candidates
    in_service = {b.connection_id for b in branches} - set(out_of_service)
    if config.type == "none":
        energised = substation_connected_component(node_ids, branches, in_service, substation_node_ids)
        return [TopologyVariant(0, set(), energised)]
    if config.type == "n_minus_k":
        return _build_n_minus_k_variants(branches, candidates, in_service, node_ids, substation_node_ids, config.k)
    if config.type == "random":
        return _build_random_variants(
            branches, candidates, in_service, node_ids, substation_node_ids, config.k, config.n_variants, rng,
        )
    raise Exception(f"Unknown topology_perturbation.type '{config.type}'.")


def build_run_variants(
    config: TopologyPerturbationConfig, configurations, branches: List[MVBranch], node_ids: List[str],
    substation_node_ids: List[str], rng: np.random.Generator, seed: int, include_transformers: bool = False,
) -> List[TopologyVariant]:
    """Every (configuration x outage set) of a run, numbered from 0. Outages
    are drawn over the MV lines closed in each configuration (plus its
    in-service transformers with include_transformers)."""
    from gridarena.pf_datagen.reconfiguration import configuration_in_service
    from gridarena.pf_datagen.sampling import STREAM_RECONFIGURATION, sample_stream

    base_in_service = {b.connection_id for b in branches if b.in_service}
    all_ids = {b.connection_id for b in branches}
    out: List[TopologyVariant] = []
    for c in configurations:
        in_service = configuration_in_service(base_in_service, c)
        candidates = [b for b in branches if b.connection_id in in_service
                      and (b.kind != "transformer" or include_transformers)]
        c_rng = rng if c.config_idx == 0 else sample_stream(seed, STREAM_RECONFIGURATION, c.config_idx)
        for v in build_variants(config, branches, node_ids, substation_node_ids, c_rng,
                                candidates=candidates, out_of_service=all_ids - in_service):
            out.append(TopologyVariant(len(out), v.disabled_connection_ids, v.energised_node_ids,
                                       c.config_idx, c.opened, c.closed))
            if len(out) > MAX_TOTAL_VARIANTS:
                raise Exception(
                    f"The run would have more than {MAX_TOTAL_VARIANTS} topology variants "
                    f"({len(configurations)} configurations x outage sets). Lower n_configs, k or n_variants."
                )
    return out


def _build_n_minus_k_variants(
    branches: List[MVBranch], candidates: List[MVBranch], all_active: Set[str], node_ids: List[str],
    substation_node_ids: List[str], k: int,
) -> List[TopologyVariant]:
    connection_ids = [b.connection_id for b in candidates]

    n_combos = sum(comb(len(connection_ids), j) for j in range(0, k + 1))
    if n_combos > MAX_N_MINUS_K_COMBOS:
        raise Exception(
            f"n_minus_k with k={k} over {len(connection_ids)} connections would enumerate "
            f"{n_combos} variants (limit {MAX_N_MINUS_K_COMBOS}). Lower k or use 'random' instead."
        )

    variants = []
    idx = 0
    for j in range(0, k + 1):
        for combo in itertools.combinations(connection_ids, j):
            disabled = set(combo)
            active = all_active - disabled
            energised = substation_connected_component(node_ids, branches, active, substation_node_ids)
            variants.append(TopologyVariant(idx, disabled, energised))
            idx += 1
    return variants


def _build_random_variants(
    branches: List[MVBranch], candidates: List[MVBranch], all_active: Set[str], node_ids: List[str],
    substation_node_ids: List[str], k: int, n_variants: int, rng: np.random.Generator,
) -> List[TopologyVariant]:
    """n_variants distinct outage sets of 1..k branches each (a repeated draw
    is redrawn, so no two variants disable the same connections). Raises if
    the grid can't supply that many distinct sets, or if the retry budget runs
    out first."""
    connection_ids = np.array([b.connection_id for b in candidates])
    k_eff = min(k, len(connection_ids))

    n_possible = sum(comb(len(connection_ids), j) for j in range(1, k_eff + 1))
    if n_variants > n_possible:
        raise Exception(
            f"random topology perturbation asked for {n_variants} distinct variants, but "
            f"{len(connection_ids)} connections with up to k={k} outages only allow {n_possible}. "
            "Lower n_variants, raise k, or use 'n_minus_k' to enumerate them all."
        )

    max_attempts = max(500, 50 * n_variants)
    seen = set()
    variants = []
    for _ in range(max_attempts):
        if len(variants) == n_variants:
            break
        r = min(int(rng.integers(1, k + 1)), len(connection_ids))
        disabled = frozenset(rng.choice(connection_ids, size=r, replace=False).tolist())
        if disabled in seen:
            continue
        seen.add(disabled)
        active = all_active - disabled
        energised = substation_connected_component(node_ids, branches, active, substation_node_ids)
        variants.append(TopologyVariant(len(variants), set(disabled), energised))
    if len(variants) < n_variants:
        raise Exception(
            f"random topology perturbation found only {len(variants)} of {n_variants} distinct variants "
            f"in {max_attempts} draws. Lower n_variants or use 'n_minus_k'."
        )
    return variants


def perturb_admittances(
    branches: List[MVBranch], sigma: float, rng: np.random.Generator,
) -> Dict[str, Tuple[float, float]]:
    """Redraw each branch's R/X uniformly in [(1-sigma)*orig, (1+sigma)*orig],
    relative to the branch's original per-km impedance (spec section 5).
    Returns {connection_id: (r_ohm_per_km, x_ohm_per_km)}. Drawn once per
    topology variant (not per scenario) -- a cable's physical impedance is a
    property of the branch itself, not of a specific timestamp. Transformers
    are skipped (their impedance isn't perturbed yet -- proposal Phase 2)."""
    lower = max(0.0, 1 - sigma)
    upper = 1 + sigma
    out = {}
    for b in branches:
        if b.kind == "transformer":
            continue
        r = rng.uniform(lower * b.r_ohm_per_km, upper * b.r_ohm_per_km)
        x = rng.uniform(lower * b.x_ohm_per_km, upper * b.x_ohm_per_km)
        out[b.connection_id] = (r, x)
    return out
