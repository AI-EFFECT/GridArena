"""DER per sample (Phase 2, axis 3).

With `der.source`:

- `grid`: only the grid's own DER, as today;
- `add`: the grid's in-service DER plus the generated plants;
- `replace`: the generated plants only.

Generated plants are listed in `der.groups` (schemas.py::DERGroup), each
group following a historical generation database
(pf_datagen/der_profiles.py) and taking its type (pv, wind, hydro, ...):

- an `aggregated` group: count_min..count_max plants of `capacity_mw` (the
  number drawn once per run from stream [seed, 19, g], so each grid of a
  group draws its own); plant k delivers
  its capacity x synthetic plant k's share at the sample's time (plants
  numbered across all groups, each its own random stream);
- an `individual` group: one plant per series of its database, rated at the
  series' maximum, delivering the series' value.

The installed capacity is therefore fixed for the run once the counts are
drawn; the penetration (installed capacity / P_max, the grid's loadability
limit) is reported, not drawn.

Each group's plants are placed by its `placement`: `random` over the
eligible nodes (MV nodes other than substations, energised in the base
topology; never a transformer's LV-side bus, which would put an MV plant
behind a small MV/LV transformer), `load` weighted by the node's transformer kVA, `clustered` around
a random centre (exp(-cable distance / cluster_km)), `feeder_end` weighted by
cable distance from the substation. Each plant takes the group's control mode,
or with `mixed` one drawn from its type's mix (schemas.py::DEFAULT_MODE_MIX).

Each sample draws from its own streams ([seed, k, sample]), so a placement
doesn't depend on the order samples are solved in and nothing is stored.
With `der.layout = per_run` every sample reuses sample 0's placement, so a
`window` timeline sees one fixed DER set-up whose output follows the
generation databases over time.
"""

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence

import numpy as np

from gridarena.pf_datagen.grid_loader import MVDER
from gridarena.pf_datagen.load_shape import cable_distances_km
from gridarena.pf_datagen.sampling import (
    STREAM_DER_COUNT, STREAM_DER_LAYOUT, STREAM_GRID_DER_SETTINGS, sample_stream,
)
from gridarena.pf_datagen.schemas import DEFAULT_MODE_MIX

FEEDER_END_FLOOR_KM = 1e-3      # keeps a node right at the substation drawable under `feeder_end`
FIXED_PF = {"fixed_pf": 1.0}


@dataclass
class DERSample:
    ders: List[MVDER]                        # the sample's DER: grid (possibly re-set) and generated
    p_available: Dict[str, float]            # der_id -> available active power (W)
    penetration: Optional[float] = None      # installed capacity / P_max; None for `grid`
    capacity_w: float = 0.0                  # installed capacity in service (VA)
    location: Optional[str] = None           # the groups' placements, e.g. "random+feeder_end"
    n_generated: int = 0
    synthetic: set = field(default_factory=set)   # der_ids of generated units


@dataclass
class Plant:
    """One generated plant: its group, rating, type and where its output comes from."""
    der_id: str
    group: int
    der_type: str
    capacity_w: float
    database_id: str
    farm: Optional[int] = None       # aggregated: synthetic plant number (its share's row in the sampler)
    series: Optional[int] = None     # individual: the series it is


def _pick(rng: np.random.Generator, mix: Dict[str, float]) -> str:
    keys = [k for k, w in mix.items() if w > 0]
    w = np.array([mix[k] for k in keys], dtype=float)
    return keys[int(rng.choice(len(keys), p=w / w.sum()))]


def draw_counts(groups, seed: int) -> List[Optional[int]]:
    """Each aggregated group's number of plants in this run, uniform in
    [count_min, count_max] (None for an individual group: its series decide)."""
    return [int(sample_stream(seed, STREAM_DER_COUNT, g).integers(group.count_min, group.count_max + 1))
            if group.interpretation == "aggregated" else None
            for g, group in enumerate(groups)]


def build_plants(groups, generation: Dict, counts: Sequence[Optional[int]]) -> List[Plant]:
    """Every generated plant of `groups` (aggregated group g: counts[g] plants),
    numbered GEN-<TYPE>-<nn> per type in group order; aggregated plants
    numbered 0, 1, ... across groups."""
    plants: List[Plant] = []
    per_type: Dict[str, int] = {}
    farm = 0
    for g, group in enumerate(groups):
        profile = generation[group.database_id]
        der_type = profile.der_type
        if group.interpretation == "aggregated":
            specs = [(group.capacity_mw * 1e6, farm + n, None) for n in range(counts[g])]
            farm += counts[g]
        else:
            # One plant per series, rated at its maximum; series that never produce are left out.
            specs = [(float(r), None, k) for k, r in enumerate(profile.ratings_w) if r > 0]
        for cap, f, k in specs:
            per_type[der_type] = per_type.get(der_type, 0) + 1
            plants.append(Plant(f"GEN-{der_type.upper()}-{per_type[der_type]:02d}", g, der_type, cap,
                                group.database_id, farm=f, series=k))
    return plants


class DERSampler:
    def __init__(self, cfg, topology, node_ids: Sequence[str], base_energised, p_max_w: Optional[float],
                 n_samples: int, seed: int, point_kva: Dict[str, float], generation: Optional[Dict] = None):
        self.cfg, self.topology, self.seed = cfg, topology, seed
        subs = set(topology.substation_node_ids)
        # MV nodes only: a transformer's LV-side bus (grid_loader.py, <transformer_id>_LV) is behind it.
        lv_side = {b.to_node_id for b in topology.branches if b.kind == "transformer"}
        self.eligible = [n for n in node_ids if n not in subs and n not in lv_side and n in base_energised]
        # Installed capacity of the grid's own DER: generating units only (an EV hub is a load).
        self.grid_capacity_w = sum(d.s_rated_kva * 1e3 for d in topology.ders
                                   if d.in_service and d.node_id in base_energised and d.der_type != "ev_hub")
        self.p_max_w = p_max_w
        # The generation databases' values at each sample (der_profiles.py), by database_id.
        self.generation = generation
        self.plants: List[Plant] = []
        self.counts: List[Optional[int]] = []
        self.farm_params = []
        self._shares: Dict[int, np.ndarray] = {}
        self._layouts: Dict[int, tuple] = {}
        if cfg.source != "grid":
            if not self.eligible:
                raise ValueError("No MV node is eligible for generated DER (energised, not a substation, not a "
                                 "transformer's LV side).")
            missing = [d for d in cfg.database_ids if not generation or d not in generation]
            if missing:
                raise ValueError(f"Generated DER need the generation profiles in the run's inputs; missing: "
                                 f"{', '.join(missing)}.")
            self.counts = draw_counts(cfg.groups, seed)
            self.plants = build_plants(cfg.groups, generation, self.counts)
            self._farm_shares()
            d = cable_distances_km(topology.branches, list(node_ids), self.eligible + sorted(subs))
            k = len(self.eligible)
            self.dist = d[:k, :k]
            to_sub = d[:k, k:].min(axis=1) if d.shape[1] > k else np.zeros(k)
            self.to_sub = np.where(np.isfinite(to_sub), to_sub, 0.0)
            self.kva = np.array([point_kva.get(n, 0.0) for n in self.eligible])

    @property
    def generated_capacity_w(self) -> float:
        return sum(p.capacity_w for p in self.plants)

    @property
    def installed_capacity_w(self) -> float:
        """Every in-service DER a generating sample has: the plants, plus the grid's own under `add`."""
        return self.generated_capacity_w + (self.grid_capacity_w if self.cfg.source == "add" else 0.0)

    @property
    def penetration(self) -> Optional[float]:
        if self.cfg.source == "grid" or not self.p_max_w:
            return None
        return self.installed_capacity_w / self.p_max_w

    # -- per sample ----------------------------------------------------------

    def sample(self, i: int, timestamp: str, grid_p_available: Dict[str, float]) -> DERSample:
        cfg = self.cfg
        grid_ders = [] if cfg.source == "replace" else list(self.topology.ders)
        if cfg.resample_grid_settings and grid_ders:
            grid_ders = self._resample_grid(i, grid_ders)
        p_av = {d.der_id: grid_p_available[d.der_id] for d in grid_ders}
        if cfg.source == "grid":
            return DERSample(grid_ders, p_av, capacity_w=self.grid_capacity_w)
        j = 0 if cfg.layout == "per_run" else i      # per_run: sample 0's placement for every sample
        units, location = self._layout(j)
        for plant, unit in zip(self.plants, units):
            if plant.farm is not None:
                p_av[unit.der_id] = plant.capacity_w * float(self._shares[plant.farm][i])
            else:
                p_av[unit.der_id] = float(self.generation[plant.database_id].values_w[plant.series, i])
        return DERSample(grid_ders + units, p_av, self.penetration, self.installed_capacity_w, location,
                         len(units), {u.der_id for u in units})

    def _layout(self, j: int):
        """The plants as units placed with sample j's draws (cached for per_run)."""
        if j in self._layouts:
            return self._layouts[j]
        cfg = self.cfg
        rng = sample_stream(self.seed, STREAM_DER_LAYOUT, j)
        units: List[MVDER] = []
        for g, group in enumerate(cfg.groups):
            plants = [p for p in self.plants if p.group == g]
            if not plants:
                continue
            weights = self._node_weights(group.placement, rng)
            if weights.sum() <= 0:
                weights = np.ones(len(self.eligible))
            for plant in plants:
                node = self.eligible[int(rng.choice(len(self.eligible), p=weights / weights.sum()))]
                mode = (_pick(rng, DEFAULT_MODE_MIX.get(plant.der_type) or FIXED_PF)
                        if group.control == "mixed" else group.control)
                cos_phi = float(rng.uniform(cfg.cos_phi_min, cfg.cos_phi_max)) if mode in ("fixed_pf", "p_v") else 1.0
                v_set = float(rng.uniform(cfg.v_set_min, cfg.v_set_max)) if mode == "voltage" else 1.02
                kva = plant.capacity_w / 1e3
                lat, lon = self.topology.node_coords.get(node, (None, None))
                units.append(MVDER(
                    der_id=plant.der_id, node_id=node, der_type=plant.der_type, s_rated_kva=kva, p_max_kw=kva,
                    profile="historical", profile_value=1.0, control_mode=mode, cos_phi=cos_phi,
                    excitation="under", v_set_pu=v_set, lat=lat, lon=lon,
                ))
        placed = {p.group for p in self.plants}
        location = "+".join(dict.fromkeys(g.placement for k, g in enumerate(cfg.groups) if k in placed)) or None
        out = (units, location)
        if cfg.layout == "per_run":
            self._layouts[j] = out
        return out

    def _farm_shares(self) -> None:
        """Each aggregated plant's output share at every sample (der_profiles.farm_shares)."""
        from gridarena.pf_datagen.der_profiles import farm_shares
        by_db: Dict[str, List[int]] = {}
        for p in self.plants:
            if p.farm is not None:
                by_db.setdefault(p.database_id, []).append(p.farm)
        for db_id, farms in by_db.items():
            shares, params = farm_shares(self.generation[db_id], farms, self.cfg.generation, self.seed)
            for f, row in zip(farms, shares):
                self._shares[f] = row
            self.farm_params.extend(params)
        self.farm_params.sort(key=lambda p: p.farm_index)

    def _node_weights(self, placement: str, rng: np.random.Generator) -> np.ndarray:
        k = len(self.eligible)
        if placement == "load":
            w = self.kva.astype(float)
            return w if w.sum() > 0 else np.ones(k)
        if placement == "clustered":
            centre = int(rng.choice(k))
            return np.exp(-self.dist[centre] / self.cfg.cluster_km)   # inf (other part) -> 0
        if placement == "feeder_end":
            return self.to_sub + FEEDER_END_FLOOR_KM
        return np.ones(k)

    def _resample_grid(self, i: int, ders: List[MVDER]) -> List[MVDER]:
        """The grid's own DER with control settings drawn from the type's mode mix."""
        cfg = self.cfg
        rng = sample_stream(self.seed, STREAM_GRID_DER_SETTINGS, i)
        out = []
        for d in ders:
            mix = DEFAULT_MODE_MIX.get(d.der_type)
            if not mix:
                out.append(d)
                continue
            mode = _pick(rng, mix)
            cos_phi = float(rng.uniform(cfg.cos_phi_min, cfg.cos_phi_max)) if mode in ("fixed_pf", "p_v") \
                else d.cos_phi
            v_set = float(rng.uniform(cfg.v_set_min, cfg.v_set_max)) if mode == "voltage" else d.v_set_pu
            out.append(replace(d, control_mode=mode, cos_phi=cos_phi, v_set_pu=v_set))
        return out
