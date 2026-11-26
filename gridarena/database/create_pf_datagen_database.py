"""Creates the sql database tables for the MV power-flow data-generation
pipeline's scenario output (bus / branch / Y-bus / runtime / summary / DER
record groups, and the measurement view's readings and meter layout).

Run metadata (config, status, progress) lives in run_config.json files under
gridarena/pf_datagen/runs/{run_id}/ -- same convention as the diffusion and RL
background jobs -- so these tables only hold the bulk per-scenario output,
keyed by an opaque run_id with no foreign key back to any run-metadata table.

Every SI value column has a per-unit twin (*_pu, see pf_datagen/per_unit.py).
Every sample holds a row for every bus/branch of the base grid, with outages
as flags (bus `energised`, branch `in_service`), and MVPFTopologyVariant lists
what each topology variant disabled -- see pf_datagen/records.py. Columns added
after the tables first shipped are NULL for rows written before they existed.
"""

_IDS = [("scenario", "INTEGER"), ("load_scenario_idx", "INTEGER")]
_F = "DOUBLE PRECISION"
_B = "BOOLEAN"

# Every column added after the tables first shipped, in the order it was
# added. CREATE TABLE below only lists the original columns; these are then
# added by _add_missing_columns (ALTER ... ADD COLUMN IF NOT EXISTS), so fresh
# and older databases end up with the same columns in the same order.
ADDED_COLUMNS = {
    "MVPFScenarioBus": [
        ("pd_pu", _F), ("qd_pu", _F), ("vm_pu", _F),                        # per-unit
        *_IDS, ("energised", _B),                                           # 3.2
        ("vm_min", _F), ("vm_max", _F), ("v_violation", _B),                # 3.3
        ("vm_lin", _F), ("vm_lin_pu", _F),                                  # 1a
        ("vn_kv", _F),                                                      # 1b
        ("pd_ref", _F), ("qd_ref", _F), ("pd_ref_pu", _F), ("qd_ref_pu", _F),  # 1c
        ("gs", _F), ("bs", _F), ("gs_pu", _F), ("bs_pu", _F),
        ("pg", _F), ("qg", _F), ("pg_pu", _F), ("qg_pu", _F),                # 1c-ii
    ],
    "MVPFScenarioBranch": [
        ("pf_pu", _F), ("qf_pu", _F), ("r_pu", _F), ("x_pu", _F),           # per-unit
        *_IDS,                                                              # 3.2
        ("pt", _F), ("qt", _F), ("p_loss", _F), ("q_loss", _F),             # 3.3
        ("i_mag", _F), ("loading_pct", _F),
        ("pt_pu", _F), ("qt_pu", _F), ("p_loss_pu", _F), ("q_loss_pu", _F),
        ("b_s_total", _F), ("b_pu", _F),                                    # 1a
        ("pf_lin", _F), ("qf_lin", _F), ("pf_lin_pu", _F), ("qf_lin_pu", _F),
        ("emergency_rating_a", _F), ("loading_emergency_pct", _F), ("thermal_violation_emergency", _B),
        ("kind", "TEXT"), ("tap", _F), ("tap_pos", "INTEGER"),              # 1b
        ("yff_r", _F), ("yff_i", _F), ("yft_r", _F), ("yft_i", _F),
        ("ytf_r", _F), ("ytf_i", _F), ("ytt_r", _F), ("ytt_i", _F),
        ("is_switch", _B),                                                  # 1d
        ("v_set_pu", _F),                                                   # 2b
        ("cable_type", "TEXT"), ("length_factor", _F),                      # 2c
    ],
    "MVPFScenarioYbus": [("g_pu", _F), ("b_pu", _F), *_IDS],
    "MVPFScenarioRuntime": [
        *_IDS,                                                              # 3.2
        ("n_iter", "INTEGER"), ("failure_reason", "TEXT"),                  # 3.4
        ("v_slack_pu", _F),                                                 # 1b
        ("control_rounds", "INTEGER"),                                      # 1c-ii
        ("solver", "TEXT"),                                                 # 1d
        ("load_level", _F), ("load_scale", _F),                             # 2a
        ("der_penetration", _F), ("der_capacity", _F), ("der_location", "TEXT"),   # 2b
        ("der_units", "INTEGER"), ("shunt_steps_in", "INTEGER"),
        ("n_der_out", "INTEGER"), ("n_shunt_out", "INTEGER"),               # 2c
        ("n_regulators_bypassed", "INTEGER"), ("conductor_temp_c", _F), ("n_cable_swaps", "INTEGER"),
        ("design_cell", "TEXT"), ("design_weight", _F),                     # 2d
    ],
    "MVPFScenarioDER": [
        ("synthetic", _B),                                                  # 2b
    ],
    "MVPFTopologyVariant": [
        ("kind", "TEXT"), ("config_idx", "INTEGER"),                        # 2c
        ("open_switches", "TEXT[]"), ("closed_ties", "TEXT[]"), ("n_disabled_transformers", "INTEGER"),
    ],
    "MVPFScenarioSummary": [
        ("p_gen", _F), ("q_gen", _F), ("p_curtailed", _F),                  # 1c-ii
        ("p_gen_pu", _F), ("q_gen_pu", _F), ("p_curtailed_pu", _F),
    ],
}


def _add_missing_columns(cursor, table):
    for col, col_type in ADDED_COLUMNS[table]:
        cursor.execute(f'ALTER TABLE "{table}" ADD COLUMN IF NOT EXISTS {col} {col_type};')


def create_mv_pf_bus_table(cursor):
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS "MVPFScenarioBus" (
            run_id TEXT NOT NULL,
            output_group_id TEXT NOT NULL,
            scenario_timestamp TEXT NOT NULL,
            topology_variant_idx INTEGER NOT NULL,
            node_id TEXT NOT NULL,
            pd DOUBLE PRECISION,
            qd DOUBLE PRECISION,
            vm DOUBLE PRECISION,
            va DOUBLE PRECISION,
            bus_type TEXT,
            PRIMARY KEY (run_id, output_group_id, node_id)
        );
        """
    )
    _add_missing_columns(cursor, "MVPFScenarioBus")
    cursor.execute('CREATE INDEX IF NOT EXISTS mvpf_bus_run_idx ON "MVPFScenarioBus"(run_id);')


def create_mv_pf_branch_table(cursor):
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS "MVPFScenarioBranch" (
            run_id TEXT NOT NULL,
            output_group_id TEXT NOT NULL,
            scenario_timestamp TEXT NOT NULL,
            topology_variant_idx INTEGER NOT NULL,
            connection_id TEXT NOT NULL,
            from_node TEXT,
            to_node TEXT,
            pf DOUBLE PRECISION,
            qf DOUBLE PRECISION,
            r_ohm_total DOUBLE PRECISION,
            x_ohm_total DOUBLE PRECISION,
            in_service BOOLEAN,
            thermal_rating_a DOUBLE PRECISION,
            thermal_violation BOOLEAN,
            PRIMARY KEY (run_id, output_group_id, connection_id)
        );
        """
    )
    _add_missing_columns(cursor, "MVPFScenarioBranch")
    cursor.execute('CREATE INDEX IF NOT EXISTS mvpf_branch_run_idx ON "MVPFScenarioBranch"(run_id);')


def create_mv_pf_ybus_table(cursor):
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS "MVPFScenarioYbus" (
            run_id TEXT NOT NULL,
            output_group_id TEXT NOT NULL,
            scenario_timestamp TEXT NOT NULL,
            topology_variant_idx INTEGER NOT NULL,
            i_node TEXT NOT NULL,
            j_node TEXT NOT NULL,
            g DOUBLE PRECISION,
            b DOUBLE PRECISION,
            PRIMARY KEY (run_id, output_group_id, i_node, j_node)
        );
        """
    )
    _add_missing_columns(cursor, "MVPFScenarioYbus")
    cursor.execute('CREATE INDEX IF NOT EXISTS mvpf_ybus_run_idx ON "MVPFScenarioYbus"(run_id);')


def create_mv_pf_runtime_table(cursor):
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS "MVPFScenarioRuntime" (
            run_id TEXT NOT NULL,
            output_group_id TEXT NOT NULL,
            scenario_timestamp TEXT NOT NULL,
            topology_variant_idx INTEGER NOT NULL,
            solve_time_ms DOUBLE PRECISION,
            converged BOOLEAN NOT NULL,
            PRIMARY KEY (run_id, output_group_id)
        );
        """
    )
    _add_missing_columns(cursor, "MVPFScenarioRuntime")
    cursor.execute('CREATE INDEX IF NOT EXISTS mvpf_runtime_run_idx ON "MVPFScenarioRuntime"(run_id);')


def create_mv_pf_variant_table(cursor):
    """One row per topology variant of a run: its configuration (the
    switches it opens and the ties it closes, relative to the base), the
    connections it disables and the nodes that leaves islanded. kind 'base'
    (configuration 0, nothing disabled) is the base topology (n_minus_k always
    includes it; random never does)."""
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS "MVPFTopologyVariant" (
            run_id TEXT NOT NULL,
            topology_variant_idx INTEGER NOT NULL,
            n_disabled INTEGER NOT NULL,
            disabled_connections TEXT[] NOT NULL,
            n_islanded INTEGER NOT NULL,
            islanded_nodes TEXT[] NOT NULL,
            PRIMARY KEY (run_id, topology_variant_idx)
        );
        """
    )
    _add_missing_columns(cursor, "MVPFTopologyVariant")


def create_mv_pf_summary_table(cursor):
    """One row per solved sample: voltage extremes and violation counts,
    highest loading, totals of load, shunt reactive power, losses and the
    power the substation supplies (see pf_datagen/records.py::build_summary_row)."""
    cursor.execute(
        f"""
        CREATE TABLE IF NOT EXISTS "MVPFScenarioSummary" (
            run_id TEXT NOT NULL,
            output_group_id TEXT NOT NULL,
            scenario_timestamp TEXT NOT NULL,
            topology_variant_idx INTEGER NOT NULL,
            scenario INTEGER,
            load_scenario_idx INTEGER,
            n_bus_energised INTEGER,
            n_bus_islanded INTEGER,
            vm_pu_min {_F},
            vm_pu_max {_F},
            n_undervoltage INTEGER,
            n_overvoltage INTEGER,
            n_branch_in_service INTEGER,
            max_loading_pct {_F},
            n_thermal_violations INTEGER,
            n_thermal_violations_emergency INTEGER,
            p_load {_F}, q_load {_F}, p_load_pu {_F}, q_load_pu {_F},
            q_shunt {_F}, q_shunt_pu {_F},
            p_loss {_F}, q_loss {_F}, p_loss_pu {_F}, q_loss_pu {_F},
            p_slack {_F}, q_slack {_F}, p_slack_pu {_F}, q_slack_pu {_F},
            PRIMARY KEY (run_id, output_group_id)
        );
        """
    )
    _add_missing_columns(cursor, "MVPFScenarioSummary")
    cursor.execute('CREATE INDEX IF NOT EXISTS mvpf_summary_run_idx ON "MVPFScenarioSummary"(run_id);')


def create_mv_pf_der_table(cursor):
    """One row per DER per solved sample: available and injected power,
    curtailment and the reactive-power bound (see
    pf_datagen/records.py::build_der_rows). Generator convention."""
    cursor.execute(
        f"""
        CREATE TABLE IF NOT EXISTS "MVPFScenarioDER" (
            run_id TEXT NOT NULL,
            output_group_id TEXT NOT NULL,
            scenario_timestamp TEXT NOT NULL,
            topology_variant_idx INTEGER NOT NULL,
            der_id TEXT NOT NULL,
            node_id TEXT,
            der_type TEXT,
            control_mode TEXT,
            in_service BOOLEAN,
            s_rated {_F}, p_available {_F}, p {_F}, q {_F}, p_curtailed {_F}, q_max {_F},
            at_q_limit BOOLEAN,
            s_rated_pu {_F}, p_available_pu {_F}, p_pu {_F}, q_pu {_F}, p_curtailed_pu {_F}, q_max_pu {_F},
            scenario INTEGER,
            load_scenario_idx INTEGER,
            PRIMARY KEY (run_id, output_group_id, der_id)
        );
        """
    )
    _add_missing_columns(cursor, "MVPFScenarioDER")
    cursor.execute('CREATE INDEX IF NOT EXISTS mvpf_der_run_idx ON "MVPFScenarioDER"(run_id);')


def create_mv_pf_measurement_tables(cursor):
    """Measurement view (Phase 2 axis 8, pf_datagen/measurements.py): one row
    per reading of a solved sample, long format, and one row per meter of a
    run's layout."""
    cursor.execute(
        f"""
        CREATE TABLE IF NOT EXISTS "MVPFMeasurement" (
            run_id TEXT NOT NULL,
            output_group_id TEXT NOT NULL,
            scenario_timestamp TEXT NOT NULL,
            topology_variant_idx INTEGER NOT NULL,
            scenario INTEGER,
            load_scenario_idx INTEGER,
            element_type TEXT NOT NULL,
            element_id TEXT NOT NULL,
            node_id TEXT,
            quantity TEXT NOT NULL,
            value {_F},
            value_pu {_F},
            source TEXT NOT NULL,
            missing BOOLEAN NOT NULL,
            is_outlier BOOLEAN NOT NULL,
            PRIMARY KEY (run_id, output_group_id, element_type, element_id, quantity)
        );
        """
    )
    cursor.execute('CREATE INDEX IF NOT EXISTS mvpf_measurement_run_idx ON "MVPFMeasurement"(run_id);')
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS "MVPFMeasurementLayout" (
            run_id TEXT NOT NULL,
            element_type TEXT NOT NULL,
            element_id TEXT NOT NULL,
            node_id TEXT,
            quantities TEXT NOT NULL,
            reason TEXT NOT NULL,
            profile TEXT,
            source TEXT NOT NULL,
            meter_end TEXT,
            PRIMARY KEY (run_id, element_type, element_id)
        );
        """
    )


def create_pf_datagen_database(conn, cursor):
    """Ensures all PF data-generation output tables exist. Idempotent via IF NOT EXISTS."""
    try:
        create_mv_pf_bus_table(cursor)
        create_mv_pf_branch_table(cursor)
        create_mv_pf_ybus_table(cursor)
        create_mv_pf_runtime_table(cursor)
        create_mv_pf_variant_table(cursor)
        create_mv_pf_summary_table(cursor)
        create_mv_pf_der_table(cursor)
        create_mv_pf_measurement_tables(cursor)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
