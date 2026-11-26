"""Column order of the MV PF data-generation output rows.

pf_datagen/records.py and job.py build each row as a tuple in this order, and
database/insert_pf_datagen_database.py inserts it with the same order (with
run_id prepended), so the two can't drift apart. Columns added later go at
the end, matching create_pf_datagen_database.ADDED_COLUMNS.
"""

_HEAD = ("output_group_id", "scenario_timestamp", "topology_variant_idx")
_IDS = ("scenario", "load_scenario_idx")

BUS = (
    *_HEAD, "node_id", "pd", "qd", "vm", "va", "bus_type", "pd_pu", "qd_pu", "vm_pu", *_IDS, "energised",
    "vm_min", "vm_max", "v_violation", "vm_lin", "vm_lin_pu", "vn_kv",
    "pd_ref", "qd_ref", "pd_ref_pu", "qd_ref_pu", "gs", "bs", "gs_pu", "bs_pu",
    "pg", "qg", "pg_pu", "qg_pu",
)

BRANCH = (
    *_HEAD, "connection_id", "from_node", "to_node", "pf", "qf", "r_ohm_total", "x_ohm_total", "in_service",
    "thermal_rating_a", "thermal_violation", "pf_pu", "qf_pu", "r_pu", "x_pu", *_IDS,
    "pt", "qt", "p_loss", "q_loss", "i_mag", "loading_pct", "pt_pu", "qt_pu", "p_loss_pu", "q_loss_pu",
    "b_s_total", "b_pu", "pf_lin", "qf_lin", "pf_lin_pu", "qf_lin_pu",
    "emergency_rating_a", "loading_emergency_pct", "thermal_violation_emergency",
    "kind", "tap", "tap_pos", "yff_r", "yff_i", "yft_r", "yft_i", "ytf_r", "ytf_i", "ytt_r", "ytt_i",
    "is_switch", "v_set_pu", "cable_type", "length_factor",
)

YBUS = (*_HEAD, "i_node", "j_node", "g", "b", "g_pu", "b_pu", *_IDS)

RUNTIME = (
    *_HEAD, "solve_time_ms", "converged", *_IDS, "n_iter", "failure_reason", "v_slack_pu", "control_rounds",
    "solver", "load_level", "load_scale",
    "der_penetration", "der_capacity", "der_location", "der_units", "shunt_steps_in",
    "n_der_out", "n_shunt_out", "n_regulators_bypassed", "conductor_temp_c", "n_cable_swaps",
    "design_cell", "design_weight",
)

SUMMARY = (
    *_HEAD, *_IDS, "n_bus_energised", "n_bus_islanded", "vm_pu_min", "vm_pu_max",
    "n_undervoltage", "n_overvoltage", "n_branch_in_service", "max_loading_pct",
    "n_thermal_violations", "n_thermal_violations_emergency",
    "p_load", "q_load", "p_load_pu", "q_load_pu", "q_shunt", "q_shunt_pu",
    "p_loss", "q_loss", "p_loss_pu", "q_loss_pu", "p_slack", "q_slack", "p_slack_pu", "q_slack_pu",
    "p_gen", "q_gen", "p_curtailed", "p_gen_pu", "q_gen_pu", "p_curtailed_pu",
)

DER = (
    *_HEAD, "der_id", "node_id", "der_type", "control_mode", "in_service",
    "s_rated", "p_available", "p", "q", "p_curtailed", "q_max", "at_q_limit",
    "s_rated_pu", "p_available_pu", "p_pu", "q_pu", "p_curtailed_pu", "q_max_pu", *_IDS,
    "synthetic",
)

MEASUREMENT = (
    *_HEAD, *_IDS, "element_type", "element_id", "node_id", "quantity", "value", "value_pu", "source",
    "missing", "is_outlier",
)

MEASUREMENT_LAYOUT = ("element_type", "element_id", "node_id", "quantities", "reason", "profile", "source",
                      "meter_end")

VARIANT = ("topology_variant_idx", "n_disabled", "disabled_connections", "n_islanded", "islanded_nodes",
           "kind", "config_idx", "open_switches", "closed_ties", "n_disabled_transformers")


def index(columns: tuple) -> dict:
    """Column name -> position in a row tuple."""
    return {c: i for i, c in enumerate(columns)}


# Column order of each table as exported (CSV export, results API, the
# standalone package's files): the sample's identifiers first.
_SAMPLE_IDS = ("scenario", "load_scenario_idx", "output_group_id", "scenario_timestamp", "topology_variant_idx")

EXPORT = {
    "bus": _SAMPLE_IDS + (
        "node_id", "energised", "pd", "qd", "vm", "va", "bus_type", "pd_pu", "qd_pu", "vm_pu",
        "vm_min", "vm_max", "v_violation", "vm_lin", "vm_lin_pu", "vn_kv",
        "pd_ref", "qd_ref", "pd_ref_pu", "qd_ref_pu", "gs", "bs", "gs_pu", "bs_pu",
        "pg", "qg", "pg_pu", "qg_pu",
    ),
    "branch": _SAMPLE_IDS + (
        "connection_id", "kind", "is_switch", "from_node", "to_node", "in_service", "tap",
        "tap_pos", "v_set_pu", "cable_type", "length_factor",
        "pf", "qf", "pt", "qt",
        "p_loss", "q_loss", "r_ohm_total", "x_ohm_total", "b_s_total", "i_mag",
        "thermal_rating_a", "loading_pct", "thermal_violation",
        "emergency_rating_a", "loading_emergency_pct", "thermal_violation_emergency",
        "pf_pu", "qf_pu", "pt_pu", "qt_pu", "p_loss_pu", "q_loss_pu", "r_pu", "x_pu", "b_pu",
        "pf_lin", "qf_lin", "pf_lin_pu", "qf_lin_pu",
        "yff_r", "yff_i", "yft_r", "yft_i", "ytf_r", "ytf_i", "ytt_r", "ytt_i",
    ),
    "ybus": _SAMPLE_IDS + ("i_node", "j_node", "g", "b", "g_pu", "b_pu"),
    "runtime": _SAMPLE_IDS + (
        "solver", "solve_time_ms", "converged", "n_iter", "failure_reason", "v_slack_pu",
        "control_rounds", "load_level", "load_scale", "der_penetration", "der_capacity",
        "der_location", "der_units", "shunt_steps_in", "n_der_out", "n_shunt_out",
        "n_regulators_bypassed", "conductor_temp_c", "n_cable_swaps",
        "design_cell", "design_weight",
    ),
    "summary": _SAMPLE_IDS + tuple(c for c in SUMMARY if c not in _SAMPLE_IDS),
    "der": _SAMPLE_IDS + tuple(c for c in DER if c not in _SAMPLE_IDS),
    "variants": VARIANT,
    "measurements": _SAMPLE_IDS + tuple(c for c in MEASUREMENT if c not in _SAMPLE_IDS),
    "layout": MEASUREMENT_LAYOUT,
}

# The row tuples each exported table is built from (records.py / job.py order).
ROWS = {"bus": BUS, "branch": BRANCH, "ybus": YBUS, "runtime": RUNTIME, "summary": SUMMARY, "der": DER,
        "variants": VARIANT, "measurements": MEASUREMENT, "layout": MEASUREMENT_LAYOUT}

# Exported table -> database table.
TABLE_NAMES = {"bus": "MVPFScenarioBus", "branch": "MVPFScenarioBranch", "ybus": "MVPFScenarioYbus",
               "runtime": "MVPFScenarioRuntime", "summary": "MVPFScenarioSummary", "der": "MVPFScenarioDER",
               "variants": "MVPFTopologyVariant", "measurements": "MVPFMeasurement",
               "layout": "MVPFMeasurementLayout"}
