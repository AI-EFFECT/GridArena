"""Measurement view (Phase 2, axis 8): next to each clean solution, the
readings a realistic set of meters would give.

**Layout.** Drawn once per run (stream [seed, 11, 0] / [seed, 11, 1]) and
stored in MVPFMeasurementLayout:
- always: each substation busbar's voltage, and P, Q and current at each
  feeder head (a branch with a substation at one end, metered at that end),
  with `substation_profile`;
- a `cp_coverage` share of the connection points: the MV-side voltage and
  the P and Q through the point's transformers (where its load sits), or the
  node's own load when it has none;
- every switch, plus a `cable_coverage` share of the other lines: P, Q and
  current at the from end;
- every DER of at least `der_min_mva`: P, Q and its node's voltage. The grid's
  own DER are fixed meters (none when `der.source` is `replace`, which drops
  them); generated DER (axis 3) change per sample, so the
  layout holds a rule (element_id `*`) and every generated unit over the size
  is metered in the samples it appears in;
- every unmetered connection point: P and Q pseudo-measurements.

**Readings**, per solved sample, from its bus, branch and DER rows:
- noise from benchmarks/general_noise's corrupt_measurement_record, called
  once per reading with a random.Random seeded by [seed, 11, 2, sample,
  variant], so readings don't depend on solving order. Current goes through
  the P/Q field, so it gets the P/Q error level. `is_outlier` is exact: the
  generator's draws are replayed from its state before the call;
- a reading is missing with probability `missing_p` ([seed, 11, 3, ...]): the
  row stays, with a null value and `missing` true. Noise is drawn for it all
  the same, so the missing rate doesn't shift the other readings' noise;
- a de-energised bus or connection point, or a branch out of service or not
  energised at both ends, reads exactly 0;
- pseudo-measurements are the true value x max(0, 1 + N(0, pseudo_sigma))
  ([seed, 11, 4, ...]), flagged `pseudo`, never missing.
Failed samples get no readings.
"""

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from gridarena.benchmarks.general_noise.noise import DEFAULT_PROFILES, corrupt_measurement_record
from gridarena.pf_datagen import columns as _columns
from gridarena.pf_datagen.sampling import STREAM_MEASUREMENTS, sample_stream

SQRT3 = 3 ** 0.5
_BUS = _columns.index(_columns.BUS)
_BRANCH = _columns.index(_columns.BRANCH)
_DER = _columns.index(_columns.DER)
_FIELD = {"v": "voltage_magnitude", "p": "power_active", "q": "power_reactive", "i": "power_active"}
SUBSTATION_REASONS = ("substation", "feeder_head")
MAX_KEPT = 50_000                 # errors kept for the report's percentiles


@dataclass(frozen=True)
class Meter:
    element_type: str             # "bus" | "branch" | "connection_point" | "der"
    element_id: str               # "*" for the generated-DER rule
    node_id: Optional[str]        # where it measures the voltage / sits (branch: its metered end)
    quantities: tuple
    reason: str                   # substation | feeder_head | switch | cable | connection_point | der |
                                  # der_generated | pseudo
    profile: Optional[str]        # None for pseudo-measurements
    end: Optional[str] = None     # branch: "from" or "to"

    @property
    def source(self) -> str:
        return "pseudo" if self.reason == "pseudo" else "meter"

    def layout_row(self) -> tuple:
        return (self.element_type, self.element_id, self.node_id, ",".join(self.quantities), self.reason,
                self.profile, self.source, self.end)


@dataclass
class MeasurementLayout:
    meters: List[Meter]
    der_min_va: float
    generated_der_profile: Optional[str]   # None: generated DER aren't metered (no rule)
    transformers_of: Dict[str, List[str]]  # connection point -> its transformers

    def rows(self) -> List[tuple]:
        return [m.layout_row() for m in self.meters]

    def counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for m in self.meters:
            if m.element_id != "*":
                out[m.reason] = out.get(m.reason, 0) + 1
        return out


def _choose(items: Sequence[str], share: float, rng: np.random.Generator) -> set:
    n = int(round(share * len(items)))
    if n <= 0:
        return set()
    return set(rng.choice(sorted(items), size=min(n, len(items)), replace=False).tolist())


def build_layout(cfg, topology, seed: int, generated_der: bool, grid_der: bool = True) -> MeasurementLayout:
    """The run's meters. `generated_der`: whether the run generates DER;
    `grid_der`: whether the grid's own DER are in its samples."""
    subs = set(topology.substation_node_ids)
    meters: List[Meter] = [Meter("bus", s, s, ("v",), "substation", cfg.substation_profile)
                           for s in topology.substation_node_ids]
    transformers_of: Dict[str, List[str]] = {}
    for b in topology.branches:
        if b.kind == "transformer":
            transformers_of.setdefault(b.from_node_id, []).append(b.connection_id)
    lines = [b for b in topology.branches if b.kind != "transformer"]
    heads = set()
    for b in lines:
        if b.from_node_id in subs or b.to_node_id in subs:
            end = "from" if b.from_node_id in subs else "to"
            meters.append(Meter("branch", b.connection_id, b.from_node_id if end == "from" else b.to_node_id,
                                ("p", "q", "i"), "feeder_head", cfg.substation_profile, end))
            heads.add(b.connection_id)

    cps = sorted(set(transformers_of) | set(topology.load_transfer))
    cps = [c for c in cps if c in topology.node_id_to_index and c not in subs]
    metered_cps = _choose(cps, cfg.cp_coverage, sample_stream(seed, STREAM_MEASUREMENTS, 0))
    others = [b.connection_id for b in lines if b.connection_id not in heads and not b.is_switch]
    metered_lines = _choose(others, cfg.cable_coverage, sample_stream(seed, STREAM_MEASUREMENTS, 1))
    for b in lines:
        if b.connection_id in heads:
            continue
        if b.is_switch or b.connection_id in metered_lines:
            meters.append(Meter("branch", b.connection_id, b.from_node_id, ("p", "q", "i"),
                                "switch" if b.is_switch else "cable", cfg.profile, "from"))
    for c in cps:
        if c in metered_cps:
            meters.append(Meter("connection_point", c, c, ("v", "p", "q"), "connection_point", cfg.profile))
    der_min_va = cfg.der_min_mva * 1e6
    for d in (topology.ders if grid_der else []):
        if d.s_rated_kva * 1e3 >= der_min_va:
            meters.append(Meter("der", d.der_id, d.node_id, ("p", "q", "v"), "der", cfg.profile))
    if generated_der:
        meters.append(Meter("der", "*", None, ("p", "q", "v"), "der_generated", cfg.profile))
    for c in cps:
        if c not in metered_cps:
            meters.append(Meter("connection_point", c, c, ("p", "q"), "pseudo", None))
    return MeasurementLayout(meters, der_min_va, cfg.profile if generated_der else None, transformers_of)


def _corrupt(x: float, quantity: str, rng: random.Random, profile: dict):
    """(noisy value, outlier?) of one reading, from corrupt_measurement_record.
    For one field it draws a Gaussian, then (with outliers on) a uniform that
    decides an outlier; replaying the state before the call tells which."""
    field = _FIELD[quantity]
    state = rng.getstate()
    noisy = corrupt_measurement_record({field: x}, rng, profile)[field]
    rel = profile["v_mag_rel_sigma"] if quantity == "v" else profile["pq_rel_sigma"]
    outlier = False
    if rel > 0 and profile["outlier_prob"] > 0 and profile["outlier_scale"] > 0:
        replay = random.Random()
        replay.setstate(state)
        replay.gauss(0.0, 1.0)
        outlier = replay.random() < profile["outlier_prob"]
    return float(noisy), outlier


class MeasurementSampler:
    """Readings of one run's layout, per solved sample."""

    def __init__(self, cfg, layout: MeasurementLayout, seed: int, bases, vn_kv_by_node: Dict[str, float]):
        self.cfg, self.layout, self.seed = cfg, layout, seed
        self.s_base = bases.s_base_va
        self.vn_kv = vn_kv_by_node
        self.profiles = {name: dict(p) for name, p in DEFAULT_PROFILES.items()}
        self.n = {"readings": 0, "missing": 0, "outliers": 0, "pseudo": 0, "dead": 0, "samples": 0}
        # |noisy - true| / |true| per profile, voltage apart from P / Q / current; outliers excluded
        self.rel_error: Dict[str, Dict[str, List[float]]] = {}
        self.pseudo_error: List[float] = []

    def _scale(self, quantity: str, node_id: str) -> float:
        """SI -> p.u. factor of a reading at node_id."""
        vn = self.vn_kv[node_id] * 1e3
        if quantity == "v":
            return 1.0 / vn
        if quantity == "i":
            return SQRT3 * vn / self.s_base
        return 1.0 / self.s_base

    def readings(self, key, variant_idx: int, bus_rows, branch_rows, der_rows) -> List[tuple]:
        bus = {r[_BUS["node_id"]]: r for r in bus_rows}
        branch = {r[_BRANCH["connection_id"]]: r for r in branch_rows}
        sample = (key.load_scenario_idx, variant_idx)
        rng = random.Random(int(sample_stream(self.seed, STREAM_MEASUREMENTS, 2, *sample).integers(2 ** 63)))
        miss_rng = sample_stream(self.seed, STREAM_MEASUREMENTS, 3, *sample)
        pseudo_rng = sample_stream(self.seed, STREAM_MEASUREMENTS, 4, *sample)
        on = lambda nid: nid in bus and bool(bus[nid][_BUS["energised"]])

        def branch_value(cid, end, q):
            r = branch[cid]
            if q == "i":
                return r[_BRANCH["i_mag"]]
            col = ("pf" if q == "p" else "qf") if end == "from" else ("pt" if q == "p" else "qt")
            return r[_BRANCH[col]]

        meters = []                                    # (meter, element_id, node, [(q, true SI)], dead)
        for m in self.layout.meters:
            if m.element_type == "bus":
                if m.node_id in bus:
                    meters.append((m, m.element_id, m.node_id, [("v", bus[m.node_id][_BUS["vm"]])],
                                   not on(m.node_id)))
            elif m.element_type == "branch":
                r = branch.get(m.element_id)
                if r is None:
                    continue
                f, t = r[_BRANCH["from_node"]], r[_BRANCH["to_node"]]
                dead = not (r[_BRANCH["in_service"]] and on(f) and on(t))
                meters.append((m, m.element_id, m.node_id,
                               [(q, branch_value(m.element_id, m.end, q)) for q in m.quantities], dead))
            elif m.element_type == "connection_point":
                if m.node_id not in bus:
                    continue
                trafos = [c for c in self.layout.transformers_of.get(m.node_id, []) if c in branch]
                if trafos:
                    p = sum(branch[c][_BRANCH["pf"]] for c in trafos)
                    q = sum(branch[c][_BRANCH["qf"]] for c in trafos)
                else:
                    p, q = bus[m.node_id][_BUS["pd"]], bus[m.node_id][_BUS["qd"]]
                vals = {"v": bus[m.node_id][_BUS["vm"]], "p": p, "q": q}
                meters.append((m, m.element_id, m.node_id, [(x, vals[x]) for x in m.quantities],
                               not on(m.node_id)))
            elif m.element_type == "der":
                rows = ([r for r in der_rows if r[_DER["synthetic"]] and r[_DER["s_rated"]] >= self.layout.der_min_va]
                        if m.element_id == "*" else [r for r in der_rows if r[_DER["der_id"]] == m.element_id])
                for r in rows:
                    nid = r[_DER["node_id"]]
                    if nid not in bus:
                        continue
                    vals = {"v": bus[nid][_BUS["vm"]], "p": r[_DER["p"]], "q": r[_DER["q"]]}
                    meters.append((m, r[_DER["der_id"]], nid, [(x, vals[x]) for x in m.quantities], not on(nid)))

        out = []
        for m, element_id, nid, values, dead in meters:
            for q, true in values:
                true = float(true or 0.0)
                scale = self._scale(q, nid)
                if m.source == "pseudo":
                    e = float(pseudo_rng.normal(0.0, self.cfg.pseudo_sigma))
                    value, missing, outlier = true * max(0.0, 1.0 + e), False, False
                    if not dead and true != 0:
                        self._keep(self.pseudo_error, abs(value - true) / abs(true))
                    if dead:
                        value = 0.0
                    self.n["pseudo"] += 1
                else:
                    value, outlier = _corrupt(true, q, rng, self.profiles[m.profile])
                    missing = bool(miss_rng.random() < self.cfg.missing_p)
                    if dead:
                        value, outlier = 0.0, False
                        self.n["dead"] += 1
                    elif not outlier and true != 0:
                        self._keep(self.rel_error.setdefault(m.profile, {}).setdefault("v" if q == "v" else "pqi", []),
                                   abs(value - true) / abs(true))
                    self.n["outliers"] += outlier
                    self.n["missing"] += missing
                    if missing:
                        value = None
                self.n["readings"] += 1
                out.append((*key.head, *key.tail, m.element_type, element_id, nid, q,
                            value, None if value is None else value * scale, m.source, missing, outlier))
        self.n["samples"] += 1
        return out

    @staticmethod
    def _keep(values: List[float], x: float) -> None:
        if len(values) < MAX_KEPT:
            values.append(x)

    def report(self) -> dict:
        def pct(v):
            if not v:
                return None
            a = np.asarray(v)
            return {f"p{p}": float(np.percentile(a, p)) for p in (10, 50, 90)} | {"n": int(a.size)}

        metered = self.n["readings"] - self.n["pseudo"]
        return {
            "profile": self.cfg.profile, "substation_profile": self.cfg.substation_profile,
            "meters": self.layout.counts(),
            "generated_der_metered": self.layout.generated_der_profile is not None,
            "samples": self.n["samples"], "readings": self.n["readings"], "pseudo_readings": self.n["pseudo"],
            "dead_readings": self.n["dead"],
            "missing_rate": self.n["missing"] / metered if metered else None,
            "outlier_rate": self.n["outliers"] / metered if metered else None,
            "relative_error": {k: {g: pct(v) for g, v in sorted(by.items())} for k, by in sorted(self.rel_error.items())},
            "pseudo_relative_error": pct(self.pseudo_error),
        }
