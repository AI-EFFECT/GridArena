"""Output rows for the MV PF data-generation pipeline.

Every solved sample writes a row for **every** bus and branch of the base
grid, whatever its topology variant switched off, so all samples of a run
share one node/edge set (what graph models need) and an outage is an explicit
flag rather than a missing row:

- bus `energised` is False for nodes the variant islanded from the
  substation: their voltage is written as 0 (vm, va, vm_pu) and their demand
  (pd, qd) is kept, so summing it gives the load the outage disconnected.
- branch `in_service` is False for connections the variant disabled (and for
  inactive connections and transformers, and open switches). Those, and in-service branches left without supply
  (both ends islanded), carry zero flow, current and loading, and no thermal
  violation.

Energised buses are checked against the run's voltage band (vm_min/vm_max,
p.u.) and flagged with v_violation; islanded buses never are -- `energised`
already says why they have no voltage.

Buses sit on their own voltage level: MV nodes on the grid's nominal voltage,
the LV side of each MV/LV transformer on its secondary voltage (vn_kv).
Branch quantities are computed in per unit (pf_datagen/solver.py) and
converted to SI with the bases of the ends they belong to: powers are
three-phase (W / VAr), currents are line currents (A), and branch impedance
and susceptance are per phase, referred to the MV side.

Lines are pi models (series impedance, half the charging susceptance at each
end), so the current differs between their two ends; i_mag and the thermal
checks use the larger one. For a transformer, i_mag is the MV-side current
and the thermal rating its rated MV-side current, so loading_pct is
essentially S / S_rated. The *_lin columns hold the LinDistFlow
approximation, as a cheap baseline.

Bus pd/qd is the demand actually drawn at the solved voltage and pd_ref/qd_ref
the demand at 1 p.u.; they are equal under the default constant-power load
model (and for islanded buses). gs/bs are the bus's shunt elements (capacitor
banks, reactors) as MATPOWER/datakit give them: the active power drawn and the
reactive power generated at 1 p.u. voltage -- bs > 0 for a capacitor. Branch
charging stays on the branch (b_s_total).

Generators (DER) are reported in the generator convention: bus pg/qg sum
the DER at the bus (positive = injected), and each DER has its own row
(build_der_rows). bus_type is "PV" while a voltage-controlling DER holds the
bus at its set-point, and "PQ" once it is at its reactive-power limit.

Each sample also gets one summary row (build_summary_row).

Row tuples follow the column order of pf_datagen/columns.py, minus the
leading run_id.
"""

from dataclasses import dataclass
from typing import AbstractSet, Dict, List, Optional, Set

import numpy as np

from gridarena.pf_datagen import columns
from gridarena.pf_datagen.solver import branch_flows_pu
from gridarena.pf_datagen.three_phase import SQRT3

_BUS = columns.index(columns.BUS)
_BRANCH = columns.index(columns.BRANCH)
_DER = columns.index(columns.DER)


@dataclass(frozen=True)
class SampleKey:
    """Identifiers shared by every row of one solved sample.

    scenario is unique per (historical timestamp x topology variant) within a
    run: load_scenario_idx * n_variants + topology_variant_idx.
    load_scenario_idx is the timestamp's position in the run's scenario list.
    """
    output_group_id: str
    timestamp: str
    topology_variant_idx: int
    scenario: int
    load_scenario_idx: int

    @property
    def head(self) -> tuple:
        return (self.output_group_id, self.timestamp, self.topology_variant_idx)

    @property
    def tail(self) -> tuple:
        return (self.scenario, self.load_scenario_idx)


def failure_reason(converged, v, error: Optional[BaseException] = None, settled: bool = True) -> Optional[str]:
    """Why a sample has no valid solution, or None if it has one:
    'solver_error: <type>: <message>' when the solver raised, 'diverged' when
    the voltages went non-finite, 'max_iterations' when it ran out of
    iterations still finite, 'control_not_settled' when voltage regulator
    taps were still moving at the round limit. The text before the first ':'
    is the category."""
    if error is not None:
        return f"solver_error: {type(error).__name__}: {error}"[:300]
    if v is None or not np.all(np.isfinite(v)):
        return "diverged"
    if not converged:
        return "max_iterations"
    if not settled:
        return "control_not_settled"
    return None


def build_bus_rows(key: SampleKey, node_ids: List[str], substation_node_ids: AbstractSet[str],
                   energised: Set[str],
                   s_by_node: Dict[str, complex], v_pu_by_node: Dict[str, complex],
                   vn_kv_by_node: Dict[str, float], bases, limits,
                   vm_lin_pu_by_node: Optional[Dict[str, float]],
                   s_drawn_by_node: Optional[Dict[str, complex]] = None,
                   y_shunt_pu_by_node: Optional[Dict[str, complex]] = None,
                   s_gen_by_node: Optional[Dict[str, complex]] = None,
                   pv_nodes: Set[str] = frozenset()) -> List[tuple]:
    """One row per node. s_by_node: three-phase demand at 1 p.u. (VA);
    s_drawn_by_node: demand drawn at the solved voltage by energised nodes
    (default: s_by_node, i.e. constant power); v_pu_by_node /
    vm_lin_pu_by_node: solved and LinDistFlow voltage (p.u. of the node's own
    level) for every energised node (vm_lin_pu_by_node None when the
    energised network isn't radial: the *_lin columns are then null);
    vn_kv_by_node: each node's line-to-line
    base voltage; limits: the run's VoltageLimits (p.u.); y_shunt_pu_by_node:
    admittance (p.u.) of each node's in-service shunt elements;
    s_gen_by_node: power injected by each node's DER (VA, generator
    convention); pv_nodes: nodes held at a set-point by a DER."""
    s_drawn_by_node = s_drawn_by_node or {}
    y_shunt_pu_by_node = y_shunt_pu_by_node or {}
    s_gen_by_node = s_gen_by_node or {}
    rows = []
    for nid in node_ids:
        s_ref = s_by_node[nid]
        vn_kv = vn_kv_by_node[nid]
        is_on = nid in energised
        s = s_drawn_by_node.get(nid, s_ref) if is_on else s_ref
        y_sh = y_shunt_pu_by_node.get(nid, 0j)
        s_gen = s_gen_by_node.get(nid, 0j) if is_on else 0j
        if is_on:
            v = v_pu_by_node[nid]
            vm_pu, va = float(abs(v)), float(np.degrees(np.angle(v)))
            vm_lin_pu = None if vm_lin_pu_by_node is None else float(vm_lin_pu_by_node[nid])
        else:
            vm_pu, va, vm_lin_pu = 0.0, 0.0, (None if vm_lin_pu_by_node is None else 0.0)
        v_base = vn_kv * 1e3
        v_violation = is_on and not (limits.vm_min <= vm_pu <= limits.vm_max)
        rows.append((
            *key.head, nid, s.real, s.imag, vm_pu * v_base, va,
            "REF" if nid in substation_node_ids else ("PV" if nid in pv_nodes else "PQ"),
            s.real / bases.s_base_va, s.imag / bases.s_base_va, vm_pu,
            *key.tail, is_on,
            limits.vm_min, limits.vm_max, bool(v_violation),
            None if vm_lin_pu is None else vm_lin_pu * v_base, vm_lin_pu,
            vn_kv,
            s_ref.real, s_ref.imag, s_ref.real / bases.s_base_va, s_ref.imag / bases.s_base_va,
            y_sh.real * bases.s_base_va, y_sh.imag * bases.s_base_va, y_sh.real, y_sh.imag,
            s_gen.real, s_gen.imag, s_gen.real / bases.s_base_va, s_gen.imag / bases.s_base_va,
        ))
    return rows


def build_branch_rows(key: SampleKey, branches, active_connection_ids: Set[str], energised: Set[str],
                      y_pu: Dict[str, complex], y_shunt_half_pu: Dict[str, complex],
                      ratio_by_branch: Dict[str, float], tap_pos_by_branch: Dict[str, Optional[int]],
                      v_pu_by_node: Dict[str, complex], vn_kv_by_node: Dict[str, float], bases,
                      s_lin_pu_by_branch: Optional[Dict[str, complex]],
                      v_set_by_branch: Optional[Dict[str, float]] = None,
                      length_factor_by_branch: Optional[Dict[str, float]] = None) -> List[tuple]:
    """One row per branch of the base grid. y_pu / y_shunt_half_pu: series
    and half shunt admittance (p.u., perturbed or not) of every branch, in
    service or not; ratio_by_branch / tap_pos_by_branch: the ratio and tap
    position used in this sample (1.0 / None for plain lines);
    s_lin_pu_by_branch: LinDistFlow power entering each energised branch at
    its from end (None when the energised network isn't radial: the *_lin
    columns are then null).

    Flows are three-phase (W / VAr): pf/qf enter at the from end, pt/qt at the
    to end (negative when power is delivered there), and the loss is their
    sum -- q_loss is negative on a lightly loaded cable, whose charging
    generates more reactive power than its reactance absorbs. loading_pct is
    i_mag / rating x 100 (None without a rating); the emergency columns
    compare i_mag with the emergency rating and are None without one.

    yff/yft/ytf/ytt_(r|i) are the branch's admittance-matrix entries in p.u.
    (as datakit reports them): Yff = y + jb/2, Yft = Ytf = -y/n,
    Ytt = y/n^2 + jb/2.

    `branches` are the branches as solved (Phase 2 axis 5 may have swapped a
    cable type, ratings included): cable_type is the type used, and
    length_factor the drawn length factor (None when lengths aren't drawn).
    """
    rows = []
    s_base, z_base = bases.s_base_va, bases.z_base_ohm
    for b in branches:
        cid = b.connection_id
        y, y_sh, n = y_pu[cid], y_shunt_half_pu.get(cid, 0j), ratio_by_branch.get(cid, 1.0)
        z = 1 / y
        in_service = cid in active_connection_ids
        if in_service and b.from_node_id in energised and b.to_node_id in energised:
            i_from, i_to, s_from, s_to = branch_flows_pu(
                v_pu_by_node[b.from_node_id], v_pu_by_node[b.to_node_id], y, y_sh, n,
            )
            s_from, s_to = s_from * s_base, s_to * s_base
            amps_from = abs(i_from) * s_base / (SQRT3 * vn_kv_by_node[b.from_node_id] * 1e3)
            amps_to = abs(i_to) * s_base / (SQRT3 * vn_kv_by_node[b.to_node_id] * 1e3)
            i_mag = amps_from if b.kind == "transformer" else max(amps_from, amps_to)
            s_lin = None if s_lin_pu_by_branch is None else s_lin_pu_by_branch[cid] * s_base
        else:
            s_from, s_to, i_mag = 0j, 0j, 0.0
            s_lin = None if s_lin_pu_by_branch is None else 0j
        s_loss = s_from + s_to
        loading_pct = 100.0 * i_mag / b.nom_curr_a if b.nom_curr_a else None
        thermal_violation = b.nom_curr_a is not None and i_mag > b.nom_curr_a
        emerg = b.emerg_curr_a
        loading_emergency_pct = 100.0 * i_mag / emerg if emerg else None
        thermal_violation_emergency = bool(i_mag > emerg) if emerg else None
        b_pu = 2 * y_sh.imag
        yff, yft, ytt = y + y_sh, -y / n, y / n ** 2 + y_sh
        rows.append((
            *key.head, cid, b.from_node_id, b.to_node_id,
            s_from.real, s_from.imag, z.real * z_base, z.imag * z_base, in_service, b.nom_curr_a,
            bool(thermal_violation),
            s_from.real / s_base, s_from.imag / s_base, z.real, z.imag,
            *key.tail,
            s_to.real, s_to.imag, s_loss.real, s_loss.imag, i_mag, loading_pct,
            s_to.real / s_base, s_to.imag / s_base, s_loss.real / s_base, s_loss.imag / s_base,
            b_pu / z_base, b_pu,
            *((None,) * 4 if s_lin is None else (s_lin.real, s_lin.imag, s_lin.real / s_base, s_lin.imag / s_base)),
            emerg, loading_emergency_pct, thermal_violation_emergency,
            b.kind, n, tap_pos_by_branch.get(cid),
            yff.real, yff.imag, yft.real, yft.imag, yft.real, yft.imag, ytt.real, ytt.imag,
            b.is_switch,
            (v_set_by_branch or {}).get(cid, b.reg_v_set_pu) if b.kind == "regulated_line" else None,
            b.cable_id,
            (length_factor_by_branch or {}).get(cid),
        ))
    return rows


def build_der_rows(key: SampleKey, ders, energised: Set[str], p_available_w: Dict[str, float],
                   p_pu: Dict[str, float], q_pu: Dict[str, float], q_max_pu: Dict[str, float],
                   at_q_limit: Dict[str, bool], bases, synthetic: AbstractSet[str] = frozenset()) -> List[tuple]:
    """One row per DER of the sample: the grid's own and any generated for it
    (der_id in `synthetic`, Phase 2 axis 3). p_available_w: what its profile
    offers this sample (W, reported even when out of service); p_pu / q_pu /
    q_max_pu / at_q_limit: the solved injection and reactive-power bound of
    each DER in service (inactive, or on an islanded node: in_service False,
    zero injection). p_curtailed is what P(V) control held back."""
    sb = bases.s_base_va
    rows = []
    for d in ders:
        on = d.in_service and d.node_id in energised and d.der_id in p_pu
        p_av = p_available_w[d.der_id]
        p = p_pu[d.der_id] * sb if on else 0.0
        q = q_pu[d.der_id] * sb if on else 0.0
        q_max = q_max_pu[d.der_id] * sb if on else 0.0
        curtailed = max(p_av - p, 0.0) if on else 0.0
        s_rated = d.s_rated_kva * 1e3
        rows.append((
            *key.head, d.der_id, d.node_id, d.der_type, d.control_mode, bool(on),
            s_rated, p_av, p, q, curtailed, q_max, bool(on and at_q_limit[d.der_id]),
            s_rated / sb, p_av / sb, p / sb, q / sb, curtailed / sb, q_max / sb,
            *key.tail,
            d.der_id in synthetic,
        ))
    return rows


def build_summary_row(key: SampleKey, bus_rows: List[tuple], branch_rows: List[tuple], bases,
                      der_rows: Optional[List[tuple]] = None) -> tuple:
    """One row summarising a solved sample from its bus, branch and DER rows:
    energised/islanded bus counts, the energised voltage range (p.u.) and the
    buses below / above their band, in-service branches, the highest loading
    and the thermal violations (normal and emergency rating), and three-phase
    totals: load drawn by energised buses, reactive power generated by shunt
    elements (capacitor banks > 0), branch losses, what the substation
    supplies -- load + losses - shunt generation - DER generation -- and the
    DER generation and curtailment."""
    on = [r for r in bus_rows if r[_BUS["energised"]]]
    vm = [r[_BUS["vm_pu"]] for r in on]
    s_load = sum((complex(r[_BUS["pd"]], r[_BUS["qd"]]) for r in on), 0j)
    p_shunt = sum(r[_BUS["gs"]] * r[_BUS["vm_pu"]] ** 2 for r in on)
    q_shunt = sum(r[_BUS["bs"]] * r[_BUS["vm_pu"]] ** 2 for r in on)
    s_loss = sum((complex(r[_BRANCH["p_loss"]], r[_BRANCH["q_loss"]]) for r in branch_rows), 0j)
    s_gen = sum((complex(r[_BUS["pg"]], r[_BUS["qg"]]) for r in on), 0j)
    p_curtailed = sum(r[_DER["p_curtailed"]] for r in der_rows or [])
    s_slack = s_load + s_loss + complex(p_shunt, -q_shunt) - s_gen
    loadings = [r[_BRANCH["loading_pct"]] for r in branch_rows if r[_BRANCH["loading_pct"]] is not None]
    sb = bases.s_base_va
    return (
        *key.head, *key.tail,
        len(on), len(bus_rows) - len(on),
        min(vm, default=None), max(vm, default=None),
        sum(r[_BUS["vm_pu"]] < r[_BUS["vm_min"]] for r in on),
        sum(r[_BUS["vm_pu"]] > r[_BUS["vm_max"]] for r in on),
        sum(bool(r[_BRANCH["in_service"]]) for r in branch_rows),
        max(loadings, default=None),
        sum(bool(r[_BRANCH["thermal_violation"]]) for r in branch_rows),
        sum(bool(r[_BRANCH["thermal_violation_emergency"]]) for r in branch_rows),
        s_load.real, s_load.imag, s_load.real / sb, s_load.imag / sb,
        q_shunt, q_shunt / sb,
        s_loss.real, s_loss.imag, s_loss.real / sb, s_loss.imag / sb,
        s_slack.real, s_slack.imag, s_slack.real / sb, s_slack.imag / sb,
        s_gen.real, s_gen.imag, p_curtailed, s_gen.real / sb, s_gen.imag / sb, p_curtailed / sb,
    )


def build_variant_rows(variants, node_ids: List[str], transformer_ids: AbstractSet[str] = frozenset()) -> List[tuple]:
    """One row per topology variant of a run: what it disables and which
    nodes that islands, its kind (base / reconfiguration / outage / both),
    configuration, the switches the configuration opens and the ties it
    closes (relative to the base), and how many transformers it disables."""
    rows = []
    for v in variants:
        disabled = sorted(v.disabled_connection_ids)
        islanded = [nid for nid in node_ids if nid not in v.energised_node_ids]
        rows.append((v.variant_idx, len(disabled), disabled, len(islanded), islanded,
                     v.kind, v.config_idx, sorted(v.opened_connection_ids), sorted(v.closed_connection_ids),
                     sum(c in transformer_ids for c in disabled)))
    return rows
