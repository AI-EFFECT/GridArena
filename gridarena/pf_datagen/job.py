"""Top-level picklable entry point for the MV PF data-generation background job.

Runs in a spawned subprocess (see pf_datagen/run_manager.py): loads its
run_config.json, builds the grid + historical scenarios once, builds the
requested topology variants once, then solves every (topology variant x
historical scenario) pair with the MV backward/forward-sweep solver (pf_datagen/solver.py),
or Newton-Raphson (pf_datagen/newton.py) when the request asks for it,
batching inserts into the MVPFScenario* tables every `chunk_commit_size`
scenarios. Each converged sample writes a row for every bus and branch of the
base grid, one row per DER and one summary row (see pf_datagen/records.py);
the variant definitions go once into MVPFTopologyVariant.

Phase 2: the scenarios' load is sized, given reactive power and noise by the
scenario builder (axis 2, pf_datagen/load_shape.py), then scaled per sample
to a load level (axis 1, pf_datagen/load_level.py), which in `loadability`
mode needs P_max from a search on the base topology. A diversity report
(pf_datagen/report.py) is accumulated during the run and written to
report.json in the run directory.
"""

import contextlib
import json
import logging
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    from threadpoolctl import threadpool_limits
except ImportError:  # optional: without it BLAS keeps its default thread count
    threadpool_limits = None

logger = logging.getLogger(__name__)


def _summary_index():
    from gridarena.pf_datagen import columns
    return columns.index(columns.SUMMARY)


def _update_run_config(run_dir: str, updates: dict) -> None:
    config_path = Path(run_dir) / "run_config.json"
    with open(config_path) as f:
        data = json.load(f)
    data.update(updates)
    with open(config_path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def run_pf_datagen(run_dir: str) -> dict:
    """Run one job. The solves work on many small matrices, where a
    multi-threaded BLAS spends far more time synchronising its threads than
    computing (Newton-Raphson ran ~18x slower on a 12-core machine), so BLAS
    is held to one thread for the duration of the job."""
    limit = threadpool_limits(limits=1) if threadpool_limits is not None else contextlib.nullcontext()
    with limit:
        return _run_pf_datagen(run_dir)


def _run_pf_datagen(run_dir: str) -> dict:
    """A GridArena run: inputs from the database, rows into it."""
    import gridarena.database as db
    from gridarena.pf_datagen.inputs import prepare_inputs
    from gridarena.pf_datagen.sinks import DatabaseSink

    run_data = start_run(run_dir)
    try:
        request, seed = request_and_seed(run_data)
        inputs = prepare_inputs(run_data["mv_grid_id"], request, seed, db)
        return execute(run_dir, run_data, request, inputs, DatabaseSink(db, run_data["run_id"]))
    except Exception as e:
        return record_failure(run_dir, e)


def start_run(run_dir: str) -> dict:
    """The run's run_config.json, marked running."""
    with open(Path(run_dir) / "run_config.json") as f:
        run_data = json.load(f)
    _update_run_config(run_dir, {"status": "running"})
    return run_data


def request_and_seed(run_data: dict):
    """The run's request, and its seed (drawn when the request has none)."""
    from gridarena.pf_datagen.schemas import PFDataGenRequest

    request = PFDataGenRequest(**{k: run_data[k] for k in PFDataGenRequest.model_fields if k in run_data})
    seed = request.seed if request.seed is not None else int(np.random.SeedSequence().entropy % (2**32))
    return request, seed


def record_failure(run_dir: str, e: Exception) -> dict:
    logger.error("PF datagen run failed: %s", e, exc_info=True)
    _update_run_config(run_dir, {
        "status": "failed",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "error": str(e),
    })
    return {"status": "failed", "error": str(e)}


def execute(run_dir: str, run_data: dict, request, inputs, sink) -> dict:
    """Solve a run from its inputs (pf_datagen/inputs.py::RunInputs) and hand
    every row to `sink` (pf_datagen/sinks.py). Needs no database: GridArena
    runs it with a DatabaseSink, the standalone package with a FileSink.
    Progress, the result and the metrics go to run_dir/run_config.json, the
    diversity report to run_dir/report.json."""
    _S_INDEX = _summary_index()
    from gridarena.pf_datagen.grid_loader import (
        branch_admittance, branch_charging_susceptance, shunt_admittance_pu, substation_connected_component,
        tap_ratio,
    )
    from gridarena.pf_datagen.der import der_control, p_available_by_scenario, run_clearness
    from gridarena.pf_datagen.der_sampling import DERSampler
    from gridarena.pf_datagen.design import Axis, Design, build_design, variant_order
    from gridarena.pf_datagen.inputs import point_kva as _point_kva
    from gridarena.pf_datagen.load_level import apply_levels, average_pattern, find_loadability, gross_demand
    from gridarena.pf_datagen.measurements import MeasurementSampler, build_layout
    from gridarena.pf_datagen.per_unit import PerUnitBases
    from gridarena.pf_datagen.outages import OutageSampler, assign_variants, bypass
    from gridarena.pf_datagen.parameters import ParameterSampler, build_catalogue, effective_branches
    from gridarena.pf_datagen.perturbation import build_run_variants, perturb_admittances
    from gridarena.pf_datagen.reconfiguration import draw_configurations
    from gridarena.pf_datagen.records import (
        SampleKey, build_branch_rows, build_bus_rows, build_der_rows, build_summary_row, build_variant_rows,
        failure_reason,
    )
    from gridarena.pf_datagen.report import DiversityReport
    from gridarena.pf_datagen.sampling import STREAM_LOAD_LEVEL, STREAM_RECONFIGURATION, draw_values, stream
    from gridarena.pf_datagen.setpoints import SetpointSampler
    from gridarena.pf_datagen.target_mix import design_weights, is_violation, pilot_ids, rounds
    from gridarena.pf_datagen.timeline import per_sample_axes, timeline_summary
    from gridarena.pf_datagen.solver import LoadModel, ShuntControl, lindistflow, solve_with_controls
    from gridarena.pf_datagen.ybus import build_ybus_entries

    run_path = Path(run_dir)
    seed, rng = inputs.seed, inputs.rng
    topology, scenarios = inputs.topology, inputs.scenarios
    # Busbar set-points come from their own streams (pf_datagen/setpoints.py),
    # so drawing them doesn't shift a run's topology/admittance draws.
    clearness_rng = np.random.default_rng([seed, 2])     # PV profile's daily clearness
    # Cable catalogue for swaps (axis 5): every grid's types, plus uploaded ones.
    catalogue = (build_catalogue(inputs.catalogue, request.parameters.catalogue)
                 if request.parameters.swap_p > 0 else [])
    node_ids = list(topology.node_id_to_index.keys())
    point_kva = _point_kva(topology)

    # A connection point's load sits behind its MV/LV transformer(s).
    for scenario in scenarios:
        S = scenario["S"]
        for cp, targets in topology.load_transfer.items():
            i_cp = topology.node_id_to_index[cp]
            s_cp, S[i_cp] = S[i_cp], 0j
            for lv_bus, share in targets:
                S[topology.node_id_to_index[lv_bus]] += share * s_cp

    # Axis 4: configurations (0 = the base) x outage sets. Without
    # reconfiguration this is the base configuration's outage sets, drawn
    # from the main generator as before.
    configurations, n_radial = draw_configurations(
        request.reconfiguration, topology.branches, node_ids, topology.substation_node_ids,
        stream(seed, STREAM_RECONFIGURATION),
    )
    variants = build_run_variants(
        request.topology_perturbation, configurations, topology.branches, node_ids,
        topology.substation_node_ids, rng, seed, include_transformers=request.outages.transformers,
    )

    # Solved per unit (pf_datagen/solver.py), each bus on its own voltage
    # level; pf_datagen/records.py converts back to SI.
    bases = PerUnitBases.for_grid(topology.nominal_voltage_kv, request.s_base_mva)
    vn_kv_by_node = {nid: topology.vn_kv(nid) for nid in node_ids}
    all_connection_ids = {b.connection_id for b in topology.branches} - topology.base_out_of_service
    # The base grid's charging, for the loadability search. admittance_perturbation
    # leaves charging alone (as in datakit, only R/X); axis 5's length factor
    # scales it with R and X (the variant loop's networks()).
    y_shunt_half_pu = {
        b.connection_id: 0.5j * branch_charging_susceptance(b) * bases.z_base_ohm for b in topology.branches
    }
    # Transformers keep their fixed tap; regulators start at theirs and move.
    start_ratio = {b.connection_id: b.ratio for b in topology.branches if b.kind != "line"}
    start_tap_pos = {b.connection_id: b.tap_pos for b in topology.branches if b.kind != "line"}
    directional = set(start_ratio)
    # Capacitor banks / reactors: fixed admittance at their node.
    y_node_shunt_all = shunt_admittance_pu(topology.shunts, bases.s_base_va)
    load_model = LoadModel(request.load_model.p.as_tuple(), request.load_model.q.as_tuple())
    # The grid's own generators: available power per historical timestamp, from
    # each one's profile, with the daily clearness drawn once for their solar
    # profiles. Generated DER follow a generation database instead (axis 3).
    der_cfg = request.der
    needs_clearness = any(d.profile == "solar" for d in topology.ders)
    clearness = run_clearness(scenarios, request.pv_clearness.min, request.pv_clearness.max,
                              clearness_rng) if needs_clearness else {}
    p_available = p_available_by_scenario(
        topology.ders, scenarios, request.timezone,
        request.pv_clearness.min, request.pv_clearness.max, clearness_rng, clearness=clearness,
    )

    # Load level per sample (Phase 2 axis 1), shared by all its variants. In
    # loadability mode the level is a fraction of P_max, found by a search
    # on the base topology with the run's average load pattern.
    level_cfg = request.load_level
    load_levels, load_scales = [None] * len(scenarios), [None] * len(scenarios)
    loadability = None
    # Demand counts only nodes the base topology energises (not load behind
    # an out-of-service transformer, which is never served).
    base_energised = substation_connected_component(
        node_ids, topology.branches, all_connection_ids, topology.substation_node_ids,
    )
    demand_mask = np.array([nid in base_energised for nid in node_ids])
    # P_max is needed for a loadability level, and reported with generated
    # DER (axis 3: their installed penetration = installed capacity / P_max).
    if level_cfg.mode == "loadability" or der_cfg.source != "grid":
        evaluate = _limit_check(topology, node_ids, request, bases, load_model, y_shunt_half_pu,
                                start_ratio, start_tap_pos, directional, y_node_shunt_all, vn_kv_by_node)
        loadability = find_loadability(average_pattern([s["S"] for s in scenarios]), evaluate,
                                       mask=demand_mask)
    # Joint design (2d): which axes pair up, and their strata.
    sp_cfg = request.setpoints
    design_axes = []
    if level_cfg.mode != "off":
        design_axes.append(Axis("level", level_cfg.min, level_cfg.max, level_cfg.bins))
    if sp_cfg.slack_mode == "uniform" and not sp_cfg.per_substation:
        design_axes.append(Axis("slack", request.slack_voltage_pu.min, request.slack_voltage_pu.max,
                                sp_cfg.slack_bins))
    design = build_design(request.design.mode, design_axes, len(scenarios), seed)
    target_cfg = request.target_mix
    targeting = target_cfg.min_violation_share is not None
    unscaled = [s["S"].copy() for s in scenarios] if targeting else None
    level_values = None
    if level_cfg.mode != "off":
        level_values = (design.values["level"].copy() if design is not None else
                        draw_values(len(scenarios), level_cfg.min, level_cfg.max, level_cfg.sampling,
                                    level_cfg.bins, stream(seed, STREAM_LOAD_LEVEL)))
        scaled, load_levels, load_scales = apply_levels(
            [s["S"] for s in scenarios], level_cfg.mode, level_values,
            loadability.p_max_w if loadability is not None else None, mask=demand_mask,
        )
        for scenario, s_scaled in zip(scenarios, scaled):
            scenario["S"] = s_scaled
    report = DiversityReport(level_cfg.mode, level_cfg.min, level_cfg.max, level_cfg.bins)

    # Axes 3 and 6: generated DER and set-points, drawn per sample (shared by
    # its variants) from streams seeded by the sample, so nothing is stored.
    setpoints = SetpointSampler(
        request.setpoints, request.slack_voltage_pu, topology.substation_node_ids, len(scenarios), seed,
        load_levels if level_cfg.mode != "off" else None,
        [gross_demand(s["S"], demand_mask) for s in scenarios],
    )
    der_sampler = DERSampler(der_cfg, topology, node_ids, base_energised,
                             loadability.p_max_w if loadability is not None else None, len(scenarios), seed,
                             point_kva, generation=getattr(inputs, "generation", None))
    # The plant groups fix the installed capacity: one penetration band, at its value.
    if der_sampler.penetration is not None:
        report.set_penetration(der_sampler.penetration)
    if design is not None and "slack" in design.values:
        setpoints.values[:, 0] = design.values["slack"]

    def axis_values():
        out = {}
        if level_values is not None:
            out["level"] = level_values
        if any(a.name == "slack" for a in design_axes):
            out["slack"] = setpoints.values[:, 0]
        return out

    # Each sample's stratum (design_cell), and the target mix's strata: load level.
    design_cells = Design(request.design.mode, design_axes).cells(axis_values()) if design_axes else None
    strata_axes = [a for a in design_axes if a.name == "level"]
    if targeting:
        pilot = pilot_ids(len(scenarios), target_cfg.pilot_share, seed)
        pilot_set = set(pilot)
        rest = [i for i in range(len(scenarios)) if i not in pilot_set]
        phases = [pilot_set] + [set(b) for b in rounds(rest, seed)]
        strata_cells = Design("", strata_axes).cells(
            {k: v for k, v in axis_values().items() if k == "level"})
    else:
        phases = [None]
    outcomes = []                     # (sample, violated or None when not converged), every output
    retarget, retargets = None, []
    shunt_mode = request.setpoints.shunt_steps
    shunts_on = [sh for sh in topology.shunts if sh.in_service]
    branch_by_id = {b.connection_id: b for b in topology.branches}
    transformers = [b for b in topology.branches if b.kind == "transformer" and b.in_service]
    transformer_ids = {b.connection_id for b in topology.branches if b.kind == "transformer"}
    regulator_ids = [b.connection_id for b in topology.branches if b.kind == "regulated_line"]
    # Axis 4: per-sample outages, and which variants each sample is solved on.
    outage_cfg = request.outages
    outage_sampler = OutageSampler(outage_cfg, seed)
    m = request.variants_per_sample
    samples_of_variant = (assign_variants(len(scenarios), len(variants), m, seed,
                                          order=variant_order(design, len(scenarios)))
                          if m != "all" and m < len(variants) else None)
    # Axis 5: branch parameters per variant or per sample.
    params = ParameterSampler(request.parameters, seed, topology.branches, catalogue)
    report.set_topology(variants, node_ids, topology.branches, topology.substation_node_ids,
                        len(configurations), n_radial)

    total = (sum(len(s) for s in samples_of_variant) if samples_of_variant is not None
             else len(scenarios) * len(variants))
    # Known now: lets the run page and the overview show progress while it runs.
    _update_run_config(run_dir, {"total_scenarios": total, "n_topology_variants": len(variants)})
    completed = 0
    failed = 0
    pending_bus, pending_branch, pending_ybus, pending_runtime, pending_summary, pending_der, pending_meas = (
        [], [], [], [], [], [], []
    )
    # Axis 8: the run's meters, and their readings per solved sample.
    meas_cfg = request.measurements
    layout = (build_layout(meas_cfg, topology, seed, der_cfg.source != "grid", der_cfg.source != "replace")
              if meas_cfg.enabled else None)
    meas = MeasurementSampler(meas_cfg, layout, seed, bases, vn_kv_by_node) if layout is not None else None

    # Drawn for every line, in service or not, so a switched-off line still
    # carries the impedance it would have in its variant.
    overrides_by_variant = [
        perturb_admittances(topology.branches, request.admittance_perturbation.sigma, rng)
        if request.admittance_perturbation.enabled else {}
        for _ in variants
    ]

    try:
        sink.variants(build_variant_rows(variants, node_ids, transformer_ids))
        if layout is not None:
            sink.layout(layout.rows())

        # Target mix (2d): a pilot first, then the rest of the samples, redrawn
        # over the strata so the run reaches its share of violating samples.
        for phase, phase_ids in enumerate(phases):
            if phase >= 1:
                solved = sum(len(p) for p in phases[:phase])
                retarget = _apply_target_mix(
                    target_cfg, strata_axes, strata_cells, solved, sorted(phase_ids),
                    sum(len(p) for p in phases[phase:]), outcomes, seed, phase, scenarios, unscaled,
                    level_cfg.mode, level_values, loadability, demand_mask, load_levels, load_scales,
                    der_sampler, setpoints,
                )
                retargets.append(retarget)
                design_cells = Design(request.design.mode, design_axes).cells(axis_values())
                strata_cells = Design("", strata_axes).cells(
                    {k: v for k, v in axis_values().items() if k == "level"})
            for variant in variants:
                active_ids = variant.active_connection_ids(all_connection_ids)
                energised = variant.energised_node_ids
                islanded_mask = demand_mask & np.array([nid not in energised for nid in node_ids])
                y_node_shunt = {nid: y for nid, y in y_node_shunt_all.items() if nid in energised}
                # MV/LV transformer off-load taps, redrawn for this variant if asked (axis 6).
                var_ratio, var_tap_pos = dict(start_ratio), dict(start_tap_pos)
                for cid, pos in setpoints.transformer_taps(variant.variant_idx, transformers).items():
                    var_ratio[cid] = tap_ratio(pos, branch_by_id[cid].tap_step_pct)
                    var_tap_pos[cid] = pos

                overrides = overrides_by_variant[variant.variant_idx]

                def networks(branches_eff, _active=active_ids, _energised=energised,
                             _ratio=var_ratio, _y_node_shunt=y_node_shunt):
                    """The networks this variant is solved on, with its branches as drawn (axis 5)."""
                    y = {b.connection_id: branch_admittance(b) * bases.z_base_ohm for b in branches_eff}
                    y_sh = {b.connection_id: 0.5j * branch_charging_susceptance(b) * bases.z_base_ohm
                            for b in branches_eff}
                    sub, lin, nw, err, regs_v = _variant_networks(
                        request.solver, topology, node_ids, _active, _energised, y, y_sh, _ratio, directional,
                        _y_node_shunt, branches=branches_eff,
                    )
                    # Capacitor banks switching their own steps (axis 6, `voltage` mode).
                    controls_sh = [
                        ShuntControl(sh.shunt_id, sh.node_id, 1j * sh.q_step_kvar * 1e3 / bases.s_base_va,
                                     sh.n_steps, sh.step, sh.step, request.setpoints.shunt_band_min,
                                     request.setpoints.shunt_band_max)
                        for sh in shunts_on if sh.node_id in _energised
                    ] if shunt_mode == "voltage" and nw is not None else []
                    # Regulators change their ratio, drawn / controlled capacitor steps
                    # and banks out their shunts, per sample: then the Y-bus is built per sample.
                    fixed = None if (regs_v or shunt_mode != "fixed" or outage_cfg.shunt_out_p > 0
                                     or params.per_sample) else build_ybus_entries(sub, y, y_sh, _ratio, _y_node_shunt)
                    return branches_eff, y, y_sh, sub, lin, nw, err, regs_v, controls_sh, fixed

                # Length, swaps and transformer impedance of this variant (axis 5);
                # with temperature or `per_sample` draws the branches change per sample.
                static_v = params.static(variant.variant_idx, None) if params.cfg.draw == "per_variant" else None
                if not params.per_sample:
                    branches_eff, length_factor, n_swaps = effective_branches(
                        topology.branches, static_v, None, overrides)
                    state = networks(branches_eff)

                sample_ids = (samples_of_variant[variant.variant_idx] if samples_of_variant is not None
                              else range(len(scenarios)))
                for load_idx in sample_ids:
                    if phase_ids is not None and load_idx not in phase_ids:
                        continue
                    scenario = scenarios[load_idx]
                    temp = params.temperature(load_idx)
                    if params.per_sample:
                        static_i = (params.static(variant.variant_idx, load_idx) if params.cfg.draw == "per_sample"
                                    else static_v)
                        branches_eff, length_factor, n_swaps = effective_branches(
                            topology.branches, static_i, temp, overrides)
                        state = networks(branches_eff)
                    (branches_eff, y_pu, y_shunt_half_v, sub_branches, lin_net, net, net_error, regulators,
                     shunt_controls, ybus_fixed) = state
                    if net_error is not None and load_idx == sample_ids[0]:
                        logger.warning("Variant %s can't be solved: %s", variant.variant_idx, net_error)
                    key = SampleKey(
                        output_group_id=f"{run_data['run_id']}-{scenario['timestamp']}-{variant.variant_idx}",
                        timestamp=scenario["timestamp"],
                        topology_variant_idx=variant.variant_idx,
                        scenario=load_idx * len(variants) + variant.variant_idx,
                        load_scenario_idx=load_idx,
                    )
                    v_slack = setpoints.slack_for_output(load_idx)
                    s_by_node = {nid: scenario["S"][topology.node_id_to_index[nid]] for nid in node_ids}
                    der_sample = der_sampler.sample(load_idx, scenario["timestamp"], p_available[load_idx])
                    # Per-sample outages (axis 4): DER trips, banks out, regulators bypassed.
                    outage = outage_sampler.sample(
                        load_idx, [d.der_id for d in der_sample.ders if d.in_service],
                        [sh.shunt_id for sh in shunts_on], regulator_ids,
                    )
                    sample_ders = [replace(d, in_service=False) if d.der_id in outage.ders_tripped else d
                                   for d in der_sample.ders]
                    active_ders = [d for d in sample_ders if d.in_service and d.node_id in energised]
                    # Installed penetration (all in-service DER / P_max) for the report's bands.
                    pen_installed = (der_sample.capacity_w / der_sampler.p_max_w
                                     if der_sample.penetration is not None else None)
                    regs = bypass(setpoints.regulators(load_idx, regulators), outage.regulators_bypassed)
                    # Capacitor steps: the grid's, drawn for the sample, then (voltage mode) as
                    # controlled; a bank that is out has none in and doesn't switch.
                    shunt_steps = {sh.shunt_id: sh.step for sh in shunts_on}
                    shunt_steps.update(setpoints.shunt_steps(load_idx, shunts_on) or {})
                    for sid in outage.shunts_out:
                        shunt_steps[sid] = 0
                    sample_shunt_controls = [c for c in shunt_controls if c.shunt_id not in outage.shunts_out]

                    # A substations-only variant (every other node islanded) has
                    # no branches and is solved trivially, with n_iter = 0.
                    error, v_pu, converged, n_iter, solve_ms = net_error, None, False, None, 0.0
                    settled, res = True, None
                    controls = [der_control(d, der_sample.p_available[d.der_id], bases.s_base_va)
                                for d in active_ders]
                    if net is not None:
                        s_pu = np.array([s_by_node[nid] for nid in net.node_ids]) / bases.s_base_va
                        t0 = time.time()
                        try:
                            res = solve_with_controls(
                                net, s_pu, v_slack, regulators=regs, ders=controls, load_model=load_model,
                                shunts=sample_shunt_controls,
                                y_shunt_extra_pu=_shunt_extra(net, shunts_on, shunt_steps, bases.s_base_va),
                            )
                            v_pu, converged, n_iter, settled = res.v, res.converged, res.n_iter, res.settled
                            shunt_steps.update(res.shunt_steps or {})
                        except Exception as e:
                            logger.warning("PF solve failed for %s: %s", key.output_group_id, e)
                            v_pu, converged, n_iter, error, res = None, False, None, e, None
                        solve_ms = (time.time() - t0) * 1000.0

                    reason = failure_reason(converged, v_pu, error, settled)
                    steps_in = sum(shunt_steps[sh.shunt_id] for sh in shunts_on
                                   if sh.node_id in energised) if shunts_on else None
                    v_slack_mean = SetpointSampler.mean(v_slack)
                    pending_runtime.append((
                        *key.head, solve_ms, reason is None, *key.tail, n_iter, reason, v_slack_mean,
                        res.rounds if res is not None else None, request.solver,
                        load_levels[load_idx], load_scales[load_idx],
                        der_sample.penetration, der_sample.capacity_w, der_sample.location,
                        der_sample.n_generated if der_cfg.source != "grid" else None, steps_in,
                        len(outage.ders_tripped) if outage_cfg.der_trip_p > 0 else None,
                        len(outage.shunts_out) if outage_cfg.shunt_out_p > 0 else None,
                        len(outage.regulators_bypassed) if outage_cfg.regulator_bypass_p > 0 else None,
                        temp.temp_c if temp is not None else None,
                        n_swaps if params.cfg.swap_p > 0 else None,
                        design_cells[load_idx] if design_cells is not None else None, None,
                    ))
                    s_arr = scenario["S"]
                    gross = float(np.real(s_arr[demand_mask])[np.real(s_arr[demand_mask]) > 0].sum())
                    p_isl = np.real(s_arr[islanded_mask])
                    report.add_topology_result(
                        variant.kind, reason, float(p_isl[p_isl > 0].sum()) / gross if gross > 0 else None,
                        temp.temp_c if temp is not None else None, n_swaps if params.cfg.swap_p > 0 else None,
                        outage if outage_sampler.enabled else None,
                    )

                    if reason is None:
                        completed += 1
                        taps = res.taps
                        ratio_by_branch, tap_pos_by_branch = dict(var_ratio), dict(var_tap_pos)
                        for r in regs:
                            ratio_by_branch[r.connection_id] = tap_ratio(taps[r.connection_id], r.tap_step_pct)
                            tap_pos_by_branch[r.connection_id] = taps[r.connection_id]
                        # DER injections (generator convention) as solved.
                        s_gen_pu = np.zeros(len(net.node_ids), dtype=complex)
                        node_pos = {nid: i for i, nid in enumerate(net.node_ids)}
                        for c in controls:
                            s_gen_pu[node_pos[c.node_id]] += complex(res.der_p[c.der_id], res.der_q[c.der_id])
                        s_gen_by_node = {nid: s_gen_pu[i] * bases.s_base_va for nid, i in node_pos.items()
                                         if s_gen_pu[i] != 0}
                        pv_nodes = {c.node_id for c in controls
                                    if c.mode == "voltage" and not res.der_at_q_limit[c.der_id]}
                        v_by_node = dict(zip(net.node_ids, v_pu))
                        s_drawn_pu = load_model.demand(s_pu, np.abs(v_pu))
                        # Constant power: the drawn demand is the reference one, exactly.
                        s_drawn_by_node = None if load_model.is_constant_power else dict(
                            zip(net.node_ids, s_drawn_pu * bases.s_base_va)
                        )
                        vm_lin_by_node, s_lin_by_branch = None, None
                        if lin_net is not None:
                            s_net = dict(zip(net.node_ids, s_drawn_pu - s_gen_pu))
                            solved_ratio = lin_net.ratios_with({r.connection_id: ratio_by_branch[r.connection_id]
                                                                for r in regs})
                            extra_lin = None
                            if res.y_shunt_extra is not None:
                                extra_by_node = dict(zip(net.node_ids, res.y_shunt_extra))
                                extra_lin = np.array([extra_by_node[nid] for nid in lin_net.node_ids])
                            vm_lin_pu, s_lin_pu = lindistflow(
                                lin_net, np.array([s_net[nid] for nid in lin_net.node_ids]), v_slack,
                                ratio=solved_ratio, y_shunt_extra_pu=extra_lin,
                            )
                            vm_lin_by_node = dict(zip(lin_net.node_ids, vm_lin_pu))
                            s_lin_by_branch = {
                                b.connection_id: lin_net.branch_sign(b.connection_id, b.from_node_id)
                                * s_lin_pu[lin_net.branch_child[b.connection_id]]
                                for b in sub_branches
                            }
                        y_shunt_sample = _shunts_with_steps(y_node_shunt_all, shunts_on, shunt_steps,
                                                            bases.s_base_va)
                        bus_rows = build_bus_rows(
                            key, node_ids, set(topology.substation_node_ids), energised, s_by_node, v_by_node,
                            vn_kv_by_node, bases, request.voltage_limits, vm_lin_by_node,
                            s_drawn_by_node=s_drawn_by_node, y_shunt_pu_by_node=y_shunt_sample,
                            s_gen_by_node=s_gen_by_node, pv_nodes=pv_nodes,
                        )
                        branch_rows = build_branch_rows(
                            key, branches_eff, active_ids, energised, y_pu, y_shunt_half_v,
                            ratio_by_branch, tap_pos_by_branch, v_by_node, vn_kv_by_node, bases, s_lin_by_branch,
                            v_set_by_branch={r.connection_id: r.v_set_pu for r in regs if r.connection_id
                                             not in outage.regulators_bypassed},
                            length_factor_by_branch=length_factor,
                        )
                        der_rows = build_der_rows(
                            key, sample_ders, energised, der_sample.p_available, res.der_p, res.der_q,
                            {c.der_id: c.q_max for c in controls}, res.der_at_q_limit, bases,
                            synthetic=der_sample.synthetic,
                        )
                        summary_row = build_summary_row(key, bus_rows, branch_rows, bases, der_rows)
                        pending_bus.extend(bus_rows)
                        pending_branch.extend(branch_rows)
                        pending_der.extend(der_rows)
                        pending_summary.append(summary_row)
                        if meas is not None:
                            pending_meas.extend(meas.readings(key, variant.variant_idx, bus_rows, branch_rows,
                                                              der_rows))
                        report.add_result(load_levels[load_idx], None, summary_row, bus_rows, branch_rows,
                                          penetration=pen_installed, v_slack=v_slack_mean,
                                          shunt_steps_in=steps_in, reg_taps=taps if regs else None)
                        report.add_kind_violations(variant.kind, summary_row)
                        if targeting:
                            outcomes.append((load_idx, is_violation(summary_row, target_cfg.violations, _S_INDEX)))
                        ybus_entries = ybus_fixed if ybus_fixed is not None else build_ybus_entries(
                            sub_branches, y_pu, y_shunt_half_v, ratio_by_branch,
                            {nid: y for nid, y in y_shunt_sample.items() if nid in energised},
                        )
                        # SI entries are referred to the MV voltage level.
                        for (i_node, j_node, g, bsus) in ybus_entries:
                            pending_ybus.append((
                                *key.head, i_node, j_node, g * bases.y_base_s, bsus * bases.y_base_s,
                                g, bsus, *key.tail,
                            ))
                    else:
                        failed += 1
                        report.add_result(load_levels[load_idx], reason, penetration=pen_installed,
                                          v_slack=v_slack_mean)
                        if targeting:
                            outcomes.append((load_idx, None))

                    pending = _maybe_flush(
                        sink, run_dir,
                        (pending_bus, pending_branch, pending_ybus, pending_runtime, pending_summary, pending_der,
                         pending_meas),
                        completed, failed, total, request.chunk_commit_size, force=False,
                    )
                    (pending_bus, pending_branch, pending_ybus, pending_runtime, pending_summary, pending_der,
                     pending_meas) = pending

        _maybe_flush(
            sink, run_dir,
            (pending_bus, pending_branch, pending_ybus, pending_runtime, pending_summary, pending_der,
             pending_meas),
            completed, failed, total, request.chunk_commit_size, force=True,
        )
        if targeting:
            weights = design_weights(strata_cells, _strata(strata_axes))
            sink.design_weights(dict(enumerate(weights)))
        sink.finish()
    finally:
        sink.close()

    # Load statistics of the samples as solved (the target mix may have rescaled some).
    for scenario in scenarios:
        report.add_load_sample(scenario["S"], demand_mask)
    if design_axes:
        report.set_design(request.design.mode, design_axes, axis_values(), design_cells,
                          design.max_rank_correlation() if design is not None else
                          Design("", design_axes, axis_values()).max_rank_correlation())
    if targeting:
        report.set_target_mix(target_cfg, retargets, outcomes, strata_cells)
    report.set_timeline({**timeline_summary(request.timeline, [s["timestamp"] for s in scenarios]),
                         "load_aggregation_s": request.load_aggregation_s},
                        per_sample_axes(request) if request.timeline == "window" else None)
    if der_cfg.source != "grid" and der_sampler.generation:
        gen_cfg = der_cfg.generation
        aggregated = [p for p in der_sampler.plants if p.farm is not None]
        params = {p.farm_index: p for p in der_sampler.farm_params}
        report.set_der_profile({
            "databases": [der_sampler.generation[d].summary() for d in der_cfg.database_ids],
            "groups": [{**g.model_dump(mode="json"), "plants": sum(1 for p in der_sampler.plants if p.group == k)}
                       for k, g in enumerate(der_cfg.groups)],
            "installed_capacity_w": der_sampler.installed_capacity_w,
            "generated_capacity_w": der_sampler.generated_capacity_w,
            "penetration": der_sampler.penetration,
            "plants": [{"der_id": p.der_id, "group": p.group, "der_type": p.der_type, "capacity_w": p.capacity_w,
                        "database_id": p.database_id,
                        **({"performance": params[p.farm].performance, "rho_step": params[p.farm].rho_step}
                           if p.farm is not None else
                           {"series_id": der_sampler.generation[p.database_id].series_ids[p.series]})}
                       for p in der_sampler.plants],
            **({"model": "synthetic plants derived from an aggregate pattern: P_k = clip(C_k f(t) a_k (1 + eps_k(t)),"
                         " 0, C_k), eps_k AR(1)",
                "rho_per_15min": gen_cfg.rho, "sigma": gen_cfg.sigma, "seed": seed} if aggregated else {}),
        })
    if meas is not None:
        report.set_measurements(meas.report())

    metrics = {
        "total_output_groups": total,
        "converged": completed,
        "non_converged": failed,
        "n_scenarios": len(scenarios),
        "n_topology_variants": len(variants),
        "n_configurations": len(configurations),
        "radial_configurations": n_radial,
        "variants_per_sample": m if samples_of_variant is not None else "all",
        "per_unit_bases": bases.as_dict(),
        "load_level_mode": level_cfg.mode,
        "timeline": request.timeline,
        "load_aggregation_s": request.load_aggregation_s,
    }
    if meas is not None:
        metrics["measurement_readings"] = meas.n["readings"]
    if loadability is not None:
        metrics.update({
            "p_max_w": loadability.p_max_w,
            "p_max_scale": loadability.scale,
            "p_max_limit": loadability.limit,
            "p_max_solves": loadability.n_solves,
        })
    with open(run_path / "report.json", "w") as f:
        json.dump(report.as_dict({"loadability": {k: metrics[k] for k in metrics if k.startswith("p_max")}
                                  if loadability is not None else None}), f)

    _update_run_config(run_dir, {
        "status": "completed",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "completed_scenarios": completed,
        "failed_scenarios": failed,
        "total_scenarios": total,
        "n_topology_variants": len(variants),
        "seed_used": seed,
        "metrics": metrics,
    })
    return {"status": "completed", "metrics": metrics}


def _strata(axes):
    """Every stratum label of `axes` (design.Design.cells format)."""
    import itertools

    return [".".join(f"{a.name[0].upper()}{b}" for a, b in zip(axes, combo))
            for combo in itertools.product(*[range(a.bins) for a in axes])]


def _apply_target_mix(cfg, axes, cells, n_solved, batch, n_remaining, outcomes, seed, round_no, scenarios,
                      unscaled, level_mode, level_values, loadability, demand_mask, load_levels, load_scales,
                      der_sampler, setpoints):
    """Before a round after the pilot: choose the strata weights from every
    output solved so far (pf_datagen/target_mix.py), redraw the round's samples'
    load level within their new strata, rescale their loads
    and follow with load-dependent set-points. Returns the target_mix.Retarget."""
    from gridarena.pf_datagen.load_level import apply_levels
    from gridarena.pf_datagen.sampling import STREAM_TARGET_MIX, sample_stream
    from gridarena.pf_datagen.target_mix import allocate, choose_weights

    strata = _strata(axes)
    rest = batch
    rt = choose_weights(strata, [(cells[i], v) for i, v in outcomes], cfg.min_violation_share, n_solved,
                        n_remaining)
    rng = sample_stream(seed, STREAM_TARGET_MIX, 1, round_no)
    labels = allocate(rt.weights, len(rest), rng)
    for i, label in zip(rest, labels):
        bins = {part[0]: int(part[1:]) for part in label.split(".")}
        for a in axes:
            v = a.lo + (bins[a.name[0].upper()] + rng.uniform()) / a.bins * (a.hi - a.lo)
            if a.name == "level":
                level_values[i] = v
    scaled, levels, scales = apply_levels(
        [unscaled[i] for i in rest], level_mode, level_values[rest],
        loadability.p_max_w if loadability is not None else None, mask=demand_mask,
    )
    for i, s_scaled, lv, sc in zip(rest, scaled, levels, scales):
        scenarios[i]["S"] = s_scaled
        load_levels[i], load_scales[i] = lv, sc
    setpoints.set_levels(rest, [load_levels[i] for i in rest])
    rt.counts = {c: labels.count(c) for c in strata}
    return rt


def _shunt_y_step(sh, s_base_va: float) -> complex:
    """Admittance (p.u.) of one step of a capacitor bank / reactor: y = jQ/S_base."""
    return 1j * sh.q_step_kvar * 1e3 / s_base_va


def _shunt_extra(net, shunts, steps, s_base_va: float):
    """Node shunt admittance (net.node_ids order) of the steps in `steps` beyond
    the grid's own, which the network already has; None when they match."""
    if not shunts or all(steps[sh.shunt_id] == sh.step for sh in shunts):
        return None
    index = {nid: i for i, nid in enumerate(net.node_ids)}
    extra = np.zeros(len(net.node_ids), dtype=complex)
    for sh in shunts:
        if sh.node_id in index:
            extra[index[sh.node_id]] += (steps[sh.shunt_id] - sh.step) * _shunt_y_step(sh, s_base_va)
    return extra


def _shunts_with_steps(y_node_shunt_all, shunts, steps, s_base_va: float):
    """Node -> shunt admittance (p.u.) with every bank at its step in `steps`."""
    out = dict(y_node_shunt_all)
    for sh in shunts:
        delta = (steps[sh.shunt_id] - sh.step) * _shunt_y_step(sh, s_base_va)
        if delta:
            out[sh.node_id] = out.get(sh.node_id, 0j) + delta
    return out


def _variant_networks(solver: str, topology, node_ids, active_ids, energised, y_pu, y_shunt_half_pu,
                      start_ratio, directional, y_node_shunt, branches=None):
    """The networks one topology variant is solved on, from `branches` (the
    grid's, or as drawn by axis 5). Returns (sub_branches, lin_net, net,
    net_error, regulators).

    Orientation/order come from the graph, not from how the connections were
    stored. With the sweep, a loop or a component fed by two substations
    fails every sample of the variant with a solver_error (net is None,
    net_error says why); Newton solves any topology. LinDistFlow needs the
    radial form, so lin_net is set only where the energised network is one."""
    from gridarena.pf_datagen.newton import MeshedNetwork, connections_on_loops
    from gridarena.pf_datagen.solver import MeshedNetworkError, RegulatorControl, build_radial_network

    sub_branches = [
        b for b in (topology.branches if branches is None else branches)
        if b.connection_id in active_ids and b.from_node_id in energised and b.to_node_id in energised
    ]
    sub_node_ids = [nid for nid in node_ids if nid in energised]
    regulators = [
        RegulatorControl(b.connection_id, b.reg_v_set_pu, b.reg_deadband_pu, b.tap_step_pct,
                         b.tap_min, b.tap_max, b.tap_pos)
        for b in sub_branches if b.kind == "regulated_line"
    ]
    roots = [s for s in topology.substation_node_ids if s in energised]
    sub_edges = [(b.connection_id, b.from_node_id, b.to_node_id) for b in sub_branches]
    sub_y_sh = {b.connection_id: y_shunt_half_pu[b.connection_id] for b in sub_branches}
    try:
        lin_net, lin_error = build_radial_network(
            sub_node_ids, roots, sub_edges,
            {b.connection_id: 1 / y_pu[b.connection_id] for b in sub_branches}, sub_y_sh,
            ratio=start_ratio, directional=directional, node_y_shunt_pu=y_node_shunt,
        ), None
    except Exception as e:
        lin_net, lin_error = None, e
    net, net_error = lin_net, lin_error
    if solver == "newton":
        try:
            net, net_error = MeshedNetwork(
                sub_node_ids, roots, sub_edges, {b.connection_id: y_pu[b.connection_id] for b in sub_branches},
                sub_y_sh, ratio=start_ratio, node_y_shunt_pu=y_node_shunt,
            ), None
        except Exception as e:
            net, net_error = None, e
        if not isinstance(lin_error, (MeshedNetworkError, type(None))):
            lin_net = None          # e.g. a ratio branch pointing towards its substation
        # A regulator on a loop holds its starting tap (see pf_datagen/newton.py).
        if regulators and lin_net is None:
            looped = connections_on_loops(sub_edges, [r.connection_id for r in regulators], roots)
            regulators = [r for r in regulators if r.connection_id not in looped]
    return sub_branches, lin_net, net, net_error, regulators


def _limit_check(topology, node_ids, request, bases, load_model, y_shunt_half_pu, start_ratio, start_tap_pos,
                 directional, y_node_shunt_all, vn_kv_by_node):
    """`evaluate(s_w)` for the loadability search (pf_datagen/load_level.py):
    solves the base topology -- unperturbed, DER off, capacitor banks as
    configured, the run's load model, the busbar at its set-point (the middle
    of its range if drawn), regulators controlling -- and returns None within
    limits, else "convergence" (no converged, settled solution), "voltage"
    (a bus below the run's vm_min) or "thermal" (a branch above 100 % of its
    normal rating, as in the output's loading_pct)."""
    from gridarena.pf_datagen import columns
    from gridarena.pf_datagen.grid_loader import branch_admittance, substation_connected_component, tap_ratio
    from gridarena.pf_datagen.load_level import LoadabilityError
    from gridarena.pf_datagen.records import SampleKey, build_branch_rows
    from gridarena.pf_datagen.schemas import SlackVoltageRange
    from gridarena.pf_datagen.solver import solve_with_controls

    active_ids = {b.connection_id for b in topology.branches} - topology.base_out_of_service
    energised = substation_connected_component(node_ids, topology.branches, active_ids, topology.substation_node_ids)
    y_pu = {b.connection_id: branch_admittance(b) * bases.z_base_ohm for b in topology.branches}
    y_node_shunt = {nid: y for nid, y in y_node_shunt_all.items() if nid in energised}
    _, _, net, net_error, regulators = _variant_networks(
        request.solver, topology, node_ids, active_ids, energised, y_pu, y_shunt_half_pu,
        start_ratio, directional, y_node_shunt,
    )
    if net is None:
        raise LoadabilityError(f"The base topology can't be solved for the loadability search: {net_error}")
    slack = request.slack_voltage_pu
    v_slack = 0.5 * (slack.min + slack.max) if isinstance(slack, SlackVoltageRange) else float(slack)
    vm_min = request.voltage_limits.vm_min
    index = topology.node_id_to_index
    key = SampleKey(output_group_id="loadability-search", timestamp="", topology_variant_idx=0,
                    scenario=0, load_scenario_idx=0)
    br = columns.index(columns.BRANCH)

    def evaluate(s_w: np.ndarray):
        s_pu = np.array([s_w[index[nid]] for nid in net.node_ids]) / bases.s_base_va
        try:
            res = solve_with_controls(net, s_pu, v_slack, regulators=regulators, load_model=load_model)
        except Exception:
            return "convergence"
        if not (res.converged and res.settled) or not np.all(np.isfinite(res.v)):
            return "convergence"
        if np.abs(res.v).min() < vm_min:
            return "voltage"
        ratio, taps = dict(start_ratio), dict(start_tap_pos)
        for r in regulators:
            ratio[r.connection_id] = tap_ratio(res.taps[r.connection_id], r.tap_step_pct)
            taps[r.connection_id] = res.taps[r.connection_id]
        rows = build_branch_rows(key, topology.branches, active_ids, energised, y_pu, y_shunt_half_pu, ratio, taps,
                                 dict(zip(net.node_ids, res.v)), vn_kv_by_node, bases, None)
        if any((row[br["loading_pct"]] or 0.0) > 100.0 for row in rows if row[br["in_service"]]):
            return "thermal"
        return None

    return evaluate


def _maybe_flush(sink, run_dir, pending, completed, failed, total, chunk_commit_size, force):
    """Hand the pending batch -- (bus, branch, ybus, runtime, summary, der,
    measurement) row lists -- to the sink and report progress once it reaches
    `chunk_commit_size` runtime rows, or unconditionally when `force=True`.
    Returns the lists to keep filling."""
    bus, branch, ybus, runtime, summary, der, meas = pending
    if not force and len(runtime) < chunk_commit_size:
        return pending

    if runtime:
        sink.batch(bus, branch, ybus, runtime, summary, der, meas)
        _update_run_config(run_dir, {"completed_scenarios": completed, "failed_scenarios": failed})
        sink.progress(completed, failed, total)

    return [], [], [], [], [], [], []
