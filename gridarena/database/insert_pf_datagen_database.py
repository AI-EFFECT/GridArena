"""Insert/query functions for the MV power-flow data-generation pipeline's
bulk scenario output (bus / branch / Y-bus / runtime / summary / DER record
groups, and the measurement view's readings and layout).

Rows are tuples in the column order of pf_datagen/columns.py."""

from gridarena.pf_datagen import columns as _columns


def _insert_rows(cursor, run_id, table, columns, rows, key):
    """executemany one INSERT per row, run_id prepended; rows already stored
    under the same key are skipped."""
    cols = ", ".join(("run_id",) + tuple(columns))
    placeholders = ", ".join(["%s"] * (len(columns) + 1))
    cursor.executemany(
        f'INSERT INTO "{table}" ({cols}) VALUES ({placeholders}) ON CONFLICT ({key}) DO NOTHING',
        [(run_id, *row) for row in rows],
    )


def insert_pf_datagen_batch(run_id, bus_rows, branch_rows, ybus_rows, runtime_rows, conn, cursor,
                            summary_rows=(), der_rows=(), measurement_rows=()):
    """Batch-insert one chunk of scenario output rows and commit.

    Each *_rows list holds plain tuples in the column order of
    pf_datagen/columns.py (run_id is prepended here so callers don't have to
    repeat it per row).
    """
    for table, columns, rows, key in (
        ("MVPFScenarioRuntime", _columns.RUNTIME, runtime_rows, "run_id, output_group_id"),
        ("MVPFScenarioBus", _columns.BUS, bus_rows, "run_id, output_group_id, node_id"),
        ("MVPFScenarioBranch", _columns.BRANCH, branch_rows, "run_id, output_group_id, connection_id"),
        ("MVPFScenarioYbus", _columns.YBUS, ybus_rows, "run_id, output_group_id, i_node, j_node"),
        ("MVPFScenarioSummary", _columns.SUMMARY, summary_rows, "run_id, output_group_id"),
        ("MVPFScenarioDER", _columns.DER, der_rows, "run_id, output_group_id, der_id"),
        ("MVPFMeasurement", _columns.MEASUREMENT, measurement_rows,
         "run_id, output_group_id, element_type, element_id, quantity"),
    ):
        if rows:
            _insert_rows(cursor, run_id, table, columns, rows, key)
    conn.commit()


def insert_pf_datagen_variants(run_id, variant_rows, conn, cursor):
    """Insert a run's topology variants (see pf_datagen/records.py::build_variant_rows) and commit."""
    if variant_rows:
        _insert_rows(cursor, run_id, "MVPFTopologyVariant", _columns.VARIANT, variant_rows,
                     "run_id, topology_variant_idx")
    conn.commit()


def insert_pf_datagen_measurement_layout(run_id, layout_rows, conn, cursor):
    """Insert a run's meter layout (pf_datagen/measurements.py) and commit."""
    if layout_rows:
        _insert_rows(cursor, run_id, "MVPFMeasurementLayout", _columns.MEASUREMENT_LAYOUT, layout_rows,
                     "run_id, element_type, element_id")
    conn.commit()


_SAMPLE_ORDER = "scenario_timestamp, topology_variant_idx"

# table key -> (SQL table name, columns returned, ORDER BY); the columns are
# the export order of pf_datagen/columns.py::EXPORT, shared with the
# standalone package's files.
_ORDER_BY = {
    "der": _SAMPLE_ORDER + ", der_id",
    "variants": "topology_variant_idx",
    "measurements": _SAMPLE_ORDER + ", element_type, element_id, quantity",
    "layout": "element_type, element_id",
}
_RESULT_TABLES = {
    key: (f'"{table}"', list(_columns.EXPORT[key]), _ORDER_BY.get(key, _SAMPLE_ORDER))
    for key, table in _columns.TABLE_NAMES.items()
}

# Postgres type -> the export's type (pyarrow names); arrays are exported
# ';'-joined, as text.
_EXPORT_TYPES = {"double precision": "float64", "integer": "int64", "boolean": "bool", "text": "string",
                 "ARRAY": "string"}


def get_pf_datagen_column_types(cursor):
    """{table key: {column: type}} of the exported tables (pf_datagen/columns.py::EXPORT),
    for typed files (the standalone package's Parquet output)."""
    cursor.execute(
        "SELECT table_name, column_name, data_type FROM information_schema.columns "
        "WHERE table_schema = current_schema() AND table_name = ANY(%s)",
        (list(_columns.TABLE_NAMES.values()),),
    )
    key_of = {t: k for k, t in _columns.TABLE_NAMES.items()}
    out = {k: {} for k in _columns.TABLE_NAMES}
    for table, column, data_type in cursor.fetchall():
        out[key_of[table]][column] = _EXPORT_TYPES.get(data_type, "string")
    return out


def get_pf_datagen_results(run_id, table, cursor, limit=500, offset=0):
    """Paginated rows for one output record group. Raises on an unknown table name."""
    if table not in _RESULT_TABLES:
        raise Exception(f"Unknown table '{table}'. Expected one of {sorted(_RESULT_TABLES)}.")
    table_name, columns, order_by = _RESULT_TABLES[table]
    cols_sql = ", ".join(columns)
    cursor.execute(
        f"SELECT {cols_sql} FROM {table_name} WHERE run_id = %s "
        f"ORDER BY {order_by} LIMIT %s OFFSET %s",
        (run_id, limit, offset),
    )
    rows = cursor.fetchall()
    return [dict(zip(columns, r)) for r in rows], columns


def set_pf_datagen_design_weights(run_id, weight_by_sample, conn, cursor):
    """Write the target mix's design_weight onto a run's runtime rows, one
    UPDATE per distinct weight ({load_scenario_idx: weight}), and commit."""
    by_weight = {}
    for idx, w in weight_by_sample.items():
        by_weight.setdefault(w, []).append(int(idx))
    for w, ids in by_weight.items():
        cursor.execute(
            'UPDATE "MVPFScenarioRuntime" SET design_weight = %s WHERE run_id = %s AND load_scenario_idx = ANY(%s)',
            (w, run_id, ids),
        )
    conn.commit()


def get_pf_datagen_split_rows(run_id, cursor):
    """(scenario, load_scenario_idx, topology_variant_idx, scenario_timestamp,
    config_idx, kind) per runtime row of a run, for split suggestions."""
    cursor.execute(
        """
        SELECT r.scenario, r.load_scenario_idx, r.topology_variant_idx, r.scenario_timestamp,
               v.config_idx, v.kind
        FROM "MVPFScenarioRuntime" r
        LEFT JOIN "MVPFTopologyVariant" v
            ON v.run_id = r.run_id AND v.topology_variant_idx = r.topology_variant_idx
        WHERE r.run_id = %s
        ORDER BY r.scenario
        """,
        (run_id,),
    )
    return cursor.fetchall()


_GROUP_STATS = {
    "vm_pu_min": "vm_pu_min",
    "vm_pu_max": "vm_pu_max",
    "max_loading_pct": "max_loading_pct",
    "p_load_w": "p_load",
    "q_load_var": "q_load",
    "loss_pct": "CASE WHEN p_slack + COALESCE(p_gen, 0) > 0 THEN 100 * p_loss / (p_slack + COALESCE(p_gen, 0)) END",
}
_PERCENTILES = (0, 5, 25, 50, 75, 95, 100)


def get_pf_datagen_group_stats(run_ids, cursor):
    """Per-sample statistics over several runs' summary rows -- the
    percentiles a run report gives, computed exactly over the whole group --
    plus, per run, its converged samples and the share with each violation."""
    out = {}
    if not run_ids:
        return {k: None for k in _GROUP_STATS}, {}
    fractions = ", ".join(str(p / 100) for p in _PERCENTILES)
    for key, expr in _GROUP_STATS.items():
        cursor.execute(
            f'SELECT percentile_cont(ARRAY[{fractions}]) WITHIN GROUP (ORDER BY {expr}), avg({expr}), count({expr}) '
            f'FROM "MVPFScenarioSummary" WHERE run_id = ANY(%s)',
            (list(run_ids),),
        )
        pct, mean, n = cursor.fetchone()
        out[key] = None if not n else {**{f"p{p}": float(v) for p, v in zip(_PERCENTILES, pct)},
                                       "mean": float(mean), "n": int(n)}
    cursor.execute(
        """
        SELECT run_id, count(*),
               count(*) FILTER (WHERE n_undervoltage > 0), count(*) FILTER (WHERE n_overvoltage > 0),
               count(*) FILTER (WHERE n_thermal_violations > 0)
        FROM "MVPFScenarioSummary" WHERE run_id = ANY(%s) GROUP BY run_id
        """,
        (list(run_ids),),
    )
    per_run = {r[0]: {"converged": r[1], "undervoltage_rate": r[2] / r[1], "overvoltage_rate": r[3] / r[1],
                      "thermal_rate": r[4] / r[1]} for r in cursor.fetchall()}
    return out, per_run


def get_pf_datagen_all_rows(run_id, table, cursor):
    """Unpaginated rows for one output record group, for CSV export. List
    values (the variants table's connection/node arrays) are joined with ';'
    so they land in one CSV cell."""
    if table not in _RESULT_TABLES:
        raise Exception(f"Unknown table '{table}'. Expected one of {sorted(_RESULT_TABLES)}.")
    table_name, columns, order_by = _RESULT_TABLES[table]
    cols_sql = ", ".join(columns)
    cursor.execute(
        f"SELECT {cols_sql} FROM {table_name} WHERE run_id = %s ORDER BY {order_by}",
        (run_id,),
    )
    rows = [tuple(";".join(v) if isinstance(v, list) else v for v in r) for r in cursor.fetchall()]
    return rows, columns


def get_pf_datagen_summary(run_id, cursor):
    """Run-wide counts and extremes for the detail page: convergence, thermal and
    voltage violations (row counts), islanded bus rows, the energised vm_pu range
    and the highest branch loading, plus non-converged samples counted by
    failure_reason category (e.g. {"max_iterations": 3}), and the mean / largest
    per-sample active loss (W) with the mean loss as a share of the power the
    substation supplies (%). The extremes and losses are None, and
    failure_reasons empty, for runs written before those columns or the
    summary table existed."""
    cursor.execute(
        'SELECT COUNT(*), COUNT(*) FILTER (WHERE converged) FROM "MVPFScenarioRuntime" WHERE run_id = %s',
        (run_id,),
    )
    total, converged = cursor.fetchone()
    cursor.execute(
        'SELECT COUNT(*) FILTER (WHERE thermal_violation), COUNT(*) FILTER (WHERE thermal_violation_emergency) '
        'FROM "MVPFScenarioBranch" WHERE run_id = %s',
        (run_id,),
    )
    violations, emergency_violations = cursor.fetchone()
    cursor.execute(
        'SELECT COUNT(*) FROM "MVPFScenarioBus" WHERE run_id = %s AND energised IS FALSE',
        (run_id,),
    )
    islanded = cursor.fetchone()[0]
    cursor.execute(
        'SELECT COUNT(*) FILTER (WHERE v_violation), MIN(vm_pu) FILTER (WHERE energised), '
        'MAX(vm_pu) FILTER (WHERE energised) FROM "MVPFScenarioBus" WHERE run_id = %s',
        (run_id,),
    )
    v_violations, vm_pu_min, vm_pu_max = cursor.fetchone()
    cursor.execute('SELECT MAX(loading_pct) FROM "MVPFScenarioBranch" WHERE run_id = %s', (run_id,))
    max_loading = cursor.fetchone()[0]
    cursor.execute(
        "SELECT split_part(failure_reason, ':', 1), COUNT(*) FROM \"MVPFScenarioRuntime\" "
        "WHERE run_id = %s AND failure_reason IS NOT NULL GROUP BY 1 ORDER BY 1",
        (run_id,),
    )
    failure_reasons = {reason: count for reason, count in cursor.fetchall()}
    cursor.execute(
        'SELECT AVG(p_loss), MAX(p_loss), AVG(p_loss / NULLIF(p_slack, 0)) FROM "MVPFScenarioSummary" '
        'WHERE run_id = %s',
        (run_id,),
    )
    p_loss_mean, p_loss_max, loss_ratio_mean = cursor.fetchone()
    return {
        "total_output_groups": total or 0,
        "converged": converged or 0,
        "thermal_violations": violations or 0,
        "thermal_violations_emergency": emergency_violations or 0,
        "voltage_violations": v_violations or 0,
        "islanded_bus_rows": islanded or 0,
        "vm_pu_min": vm_pu_min,
        "vm_pu_max": vm_pu_max,
        "max_loading_pct": max_loading,
        "failure_reasons": failure_reasons,
        "p_loss_mean": p_loss_mean,
        "p_loss_max": p_loss_max,
        "loss_pct_mean": None if loss_ratio_mean is None else 100.0 * loss_ratio_mean,
    }
