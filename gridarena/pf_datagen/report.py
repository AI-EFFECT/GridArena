"""Diversity report of a PF data-generation run (Phase 2, proposal section 5.2).

Accumulated while the job runs (no extra database reads) and stored as
report.json in the run directory; GET /pf-datagen/runs/{run_id}/report
serves it. Version 1 (part 2a) covers:

- outcomes: converged / failed samples and failure reasons;
- distributions: bus voltages (p.u.) and branch loadings (%) as histograms,
  and per-sample percentiles of the lowest / highest voltage, highest loading,
  load P / Q, losses, the load-weighted power factor and the spatial spread
  of load (coefficient of variation of |S| across load nodes);
- load level (when on): how many samples fell in each level band, how many
  converged, and how many had an undervoltage, overvoltage or thermal
  violation -- the coverage of the strata and the violation rate per band.

Version 2 (part 2b) adds:
- DER penetration (when DER are generated): samples per band of *installed*
  penetration (all in-service DER capacity / P_max -- with `add`, the grid's
  own DER count too) and, per band, the share of converged samples with
  reverse flow at the substation, an overvoltage or a thermal violation.
  The plant groups (der.groups) fix the installed capacity, so there is one
  band, at that penetration;
- the instantaneous DER/load ratio (generation / gross load, per sample);
- set-points: busbar set-point and capacitor steps switched in (percentiles),
  and how often each regulator ended at each tap.

Version 3 (part 2c) adds:
- topology: variants per kind (base / reconfiguration / outage / both), how
  many distinct energised topologies they make, nodes fed per substation
  (over variants and substations), the share of gross load islanded per
  output, and the outcomes and violation rates per variant kind;
- outages (when drawn per sample): DER units tripped, banks out and
  regulators bypassed per output;
- parameters: conductor temperature and cable swaps per output.

Version 4 (part 2d) adds:
- load: `near_duplicate_share`, the share of load samples whose injection
  vector is within about 1 % of an earlier one's (direction and magnitude
  quantised to 1 %, hashed -- approximate, but it scales to long runs);
- design: the joint design's mode and axes, the largest rank correlation
  between two axes, and how many samples fell in each stratum;
- target_mix (when on): the pilot's violation rate per stratum, the weights
  chosen, and the share of violating samples expected and reached.

Version 5 (part 2f) adds:
- timeline: `spread` or `window`, the first and last timestamp, the typical
  step and the gaps, the load aggregation interval (load_aggregation_s,
  None for the database's own step); under `window`, the settings that still draw per sample
  (they break the time continuity a forecasting dataset needs);
- measurements (when on): meters per kind, readings, and the missing and
  outlier rates and the relative error reached per noise profile (voltage
  apart from P / Q / current), plus the pseudo-measurements' relative error.
"""

import hashlib
from math import log
from typing import Dict, Iterable, List, Optional

import networkx as nx
import numpy as np

from gridarena.pf_datagen import columns
from gridarena.pf_datagen.sampling import bin_of

REPORT_VERSION = 5
VARIANT_KINDS = ("base", "reconfiguration", "outage", "both")
VM_EDGES = np.round(np.arange(0.80, 1.2001, 0.005), 3)
LOADING_EDGES = np.arange(0.0, 200.0001, 5.0)
PERCENTILES = (0, 5, 25, 50, 75, 95, 100)

_S = columns.index(columns.SUMMARY)
_BUS = columns.index(columns.BUS)
_BR = columns.index(columns.BRANCH)


class _Histogram:
    """Counts over fixed edges, plus everything below / above them."""

    def __init__(self, edges: np.ndarray):
        self.edges = edges
        self.counts = np.zeros(len(edges) - 1, dtype=np.int64)
        self.below = 0
        self.above = 0

    def add(self, values: Iterable[float]) -> None:
        v = np.asarray([x for x in values if x is not None], dtype=float)
        if v.size == 0:
            return
        self.below += int((v < self.edges[0]).sum())
        self.above += int((v > self.edges[-1]).sum())
        inside = v[(v >= self.edges[0]) & (v <= self.edges[-1])]
        self.counts += np.histogram(inside, bins=self.edges)[0]

    def as_dict(self) -> dict:
        return {"edges": [float(e) for e in self.edges], "counts": self.counts.tolist(),
                "below": self.below, "above": self.above}


def _percentiles(values: List[float]) -> Optional[Dict[str, float]]:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if v.size == 0:
        return None
    return {f"p{p}": float(np.percentile(v, p)) for p in PERCENTILES} | {"mean": float(v.mean()), "n": int(v.size)}


class DiversityReport:
    def __init__(self, load_level_mode: str = "off", level_min: Optional[float] = None,
                 level_max: Optional[float] = None, level_bins: int = 10,
                 penetration: Optional[tuple] = None):
        self.level_mode = load_level_mode
        self.level_lo, self.level_hi, self.level_bins = level_min, level_max, level_bins
        self.vm = _Histogram(VM_EDGES)
        self.loading = _Histogram(LOADING_EDGES)
        self.samples = {k: [] for k in ("vm_pu_min", "vm_pu_max", "max_loading_pct", "p_load_w", "q_load_var",
                                        "loss_pct")}
        self.loads = {k: [] for k in ("gross_demand_w", "load_pf", "spatial_cv")}
        self.outcomes = {"total": 0, "converged": 0, "failed": 0}
        self.failure_reasons: Dict[str, int] = {}
        if load_level_mode != "off":
            shape = (level_bins,)
            self.level = {k: np.zeros(shape, dtype=np.int64)
                          for k in ("samples", "converged", "undervoltage", "overvoltage", "thermal")}
            self.level_unscaled = 0
        else:
            self.level = None
        # Axis 3: penetration bands (min, max, bins) when DER are generated.
        self.pen_range = penetration
        if penetration is not None:
            self.pen = {k: np.zeros(penetration[2], dtype=np.int64)
                        for k in ("samples", "converged", "reverse_flow", "overvoltage", "thermal")}
        self.der_load_ratio: List[float] = []
        # Axis 6: set-points as solved.
        self.v_slack: List[float] = []
        self.shunt_steps: List[float] = []
        self.reg_taps: Dict[str, Dict[int, int]] = {}
        # Axes 4 and 5.
        self.topology: Optional[dict] = None
        self.by_kind = {k: {"outputs": 0, "converged": 0, "undervoltage": 0, "overvoltage": 0, "thermal": 0}
                        for k in VARIANT_KINDS}
        self.islanded_share: List[float] = []
        self.outages: Optional[Dict[str, List[int]]] = None
        self.temperature: List[float] = []
        self.cable_swaps: List[int] = []
        # 2d.
        self._load_keys = set()
        self.n_near_duplicates = 0
        self.design: Optional[dict] = None
        self.target_mix: Optional[dict] = None
        self.timeline: Optional[dict] = None
        self.der_profile: Optional[dict] = None
        self.measurements: Optional[dict] = None

    # -- accumulation ------------------------------------------------------

    def set_topology(self, variants, node_ids, branches, substation_node_ids, n_configurations: int,
                     n_radial) -> None:
        """Once, with the run's topology variants: kinds, distinct energised
        topologies and nodes fed per substation."""
        ends = {b.connection_id: (b.from_node_id, b.to_node_id) for b in branches}
        base_in_service = {b.connection_id for b in branches if b.in_service}
        subs = list(substation_node_ids)
        kinds = {k: 0 for k in VARIANT_KINDS}
        distinct = set()
        per_sub: List[float] = []
        for v in variants:
            kinds[v.kind] += 1
            energised = v.energised_node_ids
            edges = [ends[c] for c in v.active_connection_ids(base_in_service)
                     if ends[c][0] in energised and ends[c][1] in energised]
            distinct.add(frozenset(frozenset(e) for e in edges))
            g = nx.Graph()
            g.add_nodes_from(n for n in node_ids if n in energised)
            g.add_edges_from(edges)
            for comp in nx.connected_components(g):
                n_subs = sum(s in comp for s in subs)
                if n_subs:
                    per_sub.extend([len(comp) / n_subs] * n_subs)
        self.topology = {
            "n_variants": len(variants),
            "n_configurations": n_configurations,
            "radial_configurations": n_radial,
            "variants_by_kind": kinds,
            "distinct_topologies": len(distinct),
            "nodes_per_substation": _percentiles(per_sub),
        }

    def add_topology_result(self, kind: str, reason: Optional[str], islanded_share: Optional[float],
                            temperature: Optional[float], cable_swaps: Optional[int], outage=None) -> None:
        """Once per output, with add_result: its variant kind, the share of
        gross load its variant islands, its conductor temperature, cable swaps
        and per-sample outages (pf_datagen/outages.py::SampleOutage)."""
        k = self.by_kind[kind]
        k["outputs"] += 1
        k["converged"] += reason is None
        if islanded_share is not None:
            self.islanded_share.append(islanded_share)
        if temperature is not None:
            self.temperature.append(temperature)
        if cable_swaps is not None:
            self.cable_swaps.append(cable_swaps)
        if outage is not None:
            if self.outages is None:
                self.outages = {"ders_tripped": [], "shunts_out": [], "regulators_bypassed": []}
            self.outages["ders_tripped"].append(len(outage.ders_tripped))
            self.outages["shunts_out"].append(len(outage.shunts_out))
            self.outages["regulators_bypassed"].append(len(outage.regulators_bypassed))

    def add_kind_violations(self, kind: str, summary_row: tuple) -> None:
        r = summary_row
        k = self.by_kind[kind]
        k["undervoltage"] += (r[_S["n_undervoltage"]] or 0) > 0
        k["overvoltage"] += (r[_S["n_overvoltage"]] or 0) > 0
        k["thermal"] += (r[_S["n_thermal_violations"]] or 0) > 0

    def add_load_sample(self, s_w: np.ndarray, mask: Optional[np.ndarray] = None) -> None:
        """Once per load sample (before the topology variants): its injections
        in W / VAr, load convention, over the nodes in `mask` (those the base
        topology energises; all if None)."""
        s = np.asarray(s_w)
        if mask is not None:
            s = s[mask]
        p, q = np.real(s), np.imag(s)
        # Spread on magnitudes |S|: with exporting nodes the signed mean can be
        # near zero, which would make std / mean meaningless.
        mags = np.abs(s[np.abs(s) > 0])
        self.loads["gross_demand_w"].append(float(p[p > 0].sum()))
        mag = float(np.hypot(p.sum(), q.sum()))
        self.loads["load_pf"].append(float(p.sum() / mag) if mag > 0 else None)
        self.loads["spatial_cv"].append(float(mags.std() / mags.mean()) if mags.size > 1 else None)
        # Near-duplicates: direction and magnitude of the injection vector, each to ~1 %.
        norm = float(np.linalg.norm(s))
        if norm > 0:
            v = np.round(np.concatenate([p, q]) / norm * 100).astype(np.int16).tobytes()
            key = hashlib.blake2b(v + str(round(log(norm) / 0.01)).encode(), digest_size=8).digest()
            if key in self._load_keys:
                self.n_near_duplicates += 1
            else:
                self._load_keys.add(key)

    def set_design(self, mode: str, axes, values: Dict[str, np.ndarray], cells, max_rank_corr) -> None:
        """Once: the design's axes, the samples per stratum and the largest
        rank correlation between two axes (the values as solved)."""
        counts: Dict[str, int] = {}
        for c in cells or []:
            counts[c] = counts.get(c, 0) + 1
        self.design = {
            "mode": mode,
            "axes": [{"name": a.name, "min": a.lo, "max": a.hi, "bins": a.bins} for a in axes],
            "max_rank_correlation": max_rank_corr,
            "samples_per_stratum": dict(sorted(counts.items())),
            "n_strata": int(np.prod([a.bins for a in axes])),
            "strata_filled": len(counts),
        }

    def set_timeline(self, summary: dict, per_sample_axes=None) -> None:
        """The timestamps used (pf_datagen/timeline.py). Under `window`,
        per_sample_axes lists the settings that still draw per sample."""
        self.timeline = dict(summary)
        if per_sample_axes is not None:
            self.timeline["per_sample_axes"] = list(per_sample_axes)

    def set_penetration(self, penetration: float) -> None:
        """Generated DER: the installed penetration the plant groups fix -- one band at its value."""
        self.pen_range = (penetration, penetration, 1)
        self.pen = {k: np.zeros(1, dtype=np.int64)
                    for k in ("samples", "converged", "reverse_flow", "overvoltage", "thermal")}

    def set_der_profile(self, summary: dict) -> None:
        """Where the generated DER's output came from (pf_datagen/der_profiles.py)."""
        self.der_profile = summary

    def set_measurements(self, summary: dict) -> None:
        """The measurement view's layout counts and realised rates (pf_datagen/measurements.py)."""
        self.measurements = summary

    def set_target_mix(self, cfg, retargets, outcomes, cells) -> None:
        """Once: the target mix's rounds (target_mix.Retarget each), the last
        round's rates and weights, and the result. outcomes: (sample, violated
        or None) per output."""
        retarget = retargets[-1] if retargets else None
        converged = [v for _, v in outcomes if v is not None]
        out = {
            "min_violation_share": cfg.min_violation_share,
            "violations": cfg.violations,
            "pilot_share": cfg.pilot_share,
            "achieved_share": (sum(converged) / len(converged)) if converged else None,
            "converged_outputs": len(converged),
            "rounds": [{"tilt": r.tilt, "expected_share": r.expected_share, "reachable": r.reachable}
                       for r in retargets],
        }
        if retarget is not None:
            out.update({
                "tilt": retarget.tilt,
                "reachable": retarget.reachable,
                "expected_share": retarget.expected_share,
                "rate": retarget.pilot_rate,
                "outputs": retarget.pilot_outputs,
                "weights": retarget.weights,
                "last_round_samples_per_stratum": retarget.counts,
            })
        self.target_mix = out

    def add_result(self, level: Optional[float], reason: Optional[str], summary_row: Optional[tuple] = None,
                   bus_rows: Iterable[tuple] = (), branch_rows: Iterable[tuple] = (),
                   penetration: Optional[float] = None, v_slack: Optional[float] = None,
                   shunt_steps_in: Optional[int] = None, reg_taps: Optional[Dict[str, int]] = None) -> None:
        """Once per solved output (sample x variant)."""
        self.outcomes["total"] += 1
        if v_slack is not None:
            self.v_slack.append(v_slack)
        if reason is None and shunt_steps_in is not None:
            self.shunt_steps.append(shunt_steps_in)
        for cid, tap in (reg_taps or {}).items():
            counts = self.reg_taps.setdefault(cid, {})
            counts[int(tap)] = counts.get(int(tap), 0) + 1
        bus_rows = list(bus_rows)
        self._add_penetration(penetration, reason, summary_row, bus_rows)
        if reason is None:
            self.outcomes["converged"] += 1
        else:
            self.outcomes["failed"] += 1
            key = reason.split(":")[0]
            self.failure_reasons[key] = self.failure_reasons.get(key, 0) + 1

        under = over = thermal = False
        if summary_row is not None:
            r = summary_row
            self.samples["vm_pu_min"].append(r[_S["vm_pu_min"]])
            self.samples["vm_pu_max"].append(r[_S["vm_pu_max"]])
            self.samples["max_loading_pct"].append(r[_S["max_loading_pct"]])
            self.samples["p_load_w"].append(r[_S["p_load"]])
            self.samples["q_load_var"].append(r[_S["q_load"]])
            p_in = (r[_S["p_slack"]] or 0.0) + (r[_S["p_gen"]] or 0.0)
            self.samples["loss_pct"].append(100.0 * r[_S["p_loss"]] / p_in if p_in > 0 else None)
            under = (r[_S["n_undervoltage"]] or 0) > 0
            over = (r[_S["n_overvoltage"]] or 0) > 0
            thermal = (r[_S["n_thermal_violations"]] or 0) > 0
            self.vm.add(b[_BUS["vm_pu"]] for b in bus_rows if b[_BUS["energised"]])
            self.loading.add(b[_BR["loading_pct"]] for b in branch_rows if b[_BR["in_service"]])

        if self.level is not None:
            if level is None:
                self.level_unscaled += 1
                return
            i = int(bin_of(np.array([level]), self.level_lo, self.level_hi, self.level_bins)[0])
            self.level["samples"][i] += 1
            if reason is None:
                self.level["converged"][i] += 1
                self.level["undervoltage"][i] += under
                self.level["overvoltage"][i] += over
                self.level["thermal"][i] += thermal

    def _add_penetration(self, penetration, reason, summary_row, bus_rows) -> None:
        """penetration: the sample's installed DER capacity / P_max."""
        if summary_row is not None:
            # Generation over gross load (positive demand of energised buses): the
            # net load can be near zero when loads export, making a ratio to it meaningless.
            gross = sum(b[_BUS["pd"]] for b in bus_rows if b[_BUS["energised"]] and b[_BUS["pd"]] > 0)
            self.der_load_ratio.append((summary_row[_S["p_gen"]] or 0.0) / gross if gross > 0 else None)
        if self.pen_range is None or penetration is None:
            return
        lo, hi, bins = self.pen_range
        i = int(bin_of(np.array([penetration]), lo, hi, bins)[0])
        self.pen["samples"][i] += 1
        if reason is None and summary_row is not None:
            r = summary_row
            self.pen["converged"][i] += 1
            self.pen["reverse_flow"][i] += (r[_S["p_slack"]] or 0.0) < 0
            self.pen["overvoltage"][i] += (r[_S["n_overvoltage"]] or 0) > 0
            self.pen["thermal"][i] += (r[_S["n_thermal_violations"]] or 0) > 0

    # -- output ------------------------------------------------------------

    def as_dict(self, extra: Optional[dict] = None) -> dict:
        out = {
            "version": REPORT_VERSION,
            "outcomes": self.outcomes,
            "failure_reasons": self.failure_reasons,
            "voltage_pu": self.vm.as_dict(),
            "loading_pct": self.loading.as_dict(),
            "per_sample": {k: _percentiles(v) for k, v in self.samples.items()},
            "load": {k: _percentiles(v) for k, v in self.loads.items()},
        }
        n_loads = len(self.loads["gross_demand_w"])
        out["load"]["near_duplicate_share"] = self.n_near_duplicates / n_loads if n_loads else None
        if self.design is not None:
            out["design"] = self.design
        if self.target_mix is not None:
            out["target_mix"] = self.target_mix
        if self.timeline is not None:
            out["timeline"] = self.timeline
        if self.der_profile is not None:
            out["der_profile"] = self.der_profile
        if self.measurements is not None:
            out["measurements"] = self.measurements
        if self.level is not None:
            edges = np.linspace(self.level_lo, self.level_hi, self.level_bins + 1)
            conv = self.level["converged"]
            rate = (lambda k: [float(x / c) if c else None for x, c in zip(self.level[k], conv)])
            out["load_level"] = {
                "mode": self.level_mode,
                "edges": [float(e) for e in edges],
                "samples": self.level["samples"].tolist(),
                "converged": conv.tolist(),
                "undervoltage_rate": rate("undervoltage"),
                "overvoltage_rate": rate("overvoltage"),
                "thermal_rate": rate("thermal"),
                "unscaled_samples": self.level_unscaled,
            }
        if self.pen_range is not None:
            lo, hi, bins = self.pen_range
            conv = self.pen["converged"]
            prate = (lambda k: [float(x / c) if c else None for x, c in zip(self.pen[k], conv)])
            out["der_penetration"] = {
                "edges": [float(e) for e in np.linspace(lo, hi, bins + 1)],
                "samples": self.pen["samples"].tolist(),
                "converged": conv.tolist(),
                "reverse_flow_rate": prate("reverse_flow"),
                "overvoltage_rate": prate("overvoltage"),
                "thermal_rate": prate("thermal"),
            }
        out["der_load_ratio"] = _percentiles(self.der_load_ratio)
        out["setpoints"] = {
            "v_slack_pu": _percentiles(self.v_slack),
            "shunt_steps_in": _percentiles(self.shunt_steps),
            "regulator_taps": {cid: dict(sorted(c.items())) for cid, c in sorted(self.reg_taps.items())},
        }
        if self.topology is not None:
            kinds = {}
            for name, k in self.by_kind.items():
                if not k["outputs"]:
                    continue
                c = k["converged"]
                kinds[name] = {"outputs": k["outputs"], "converged": c,
                               **{f"{v}_rate": (k[v] / c if c else None)
                                  for v in ("undervoltage", "overvoltage", "thermal")}}
            share = [x for x in self.islanded_share if x is not None]
            out["topology"] = {
                **self.topology,
                "outcomes_by_kind": kinds,
                "islanded_load_share": _percentiles(share),
                "outputs_with_islanded_load": int(sum(x > 0 for x in share)),
            }
        if self.outages is not None:
            out["outages"] = {k: _percentiles(v) for k, v in self.outages.items()}
        if self.temperature or self.cable_swaps:
            out["parameters"] = {
                "conductor_temp_c": _percentiles(self.temperature),
                "cable_swaps": _percentiles(self.cable_swaps),
            }
        if extra:
            out.update(extra)
        return out
