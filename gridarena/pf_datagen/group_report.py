"""Report of a run group (Phase 2, part 2d; axis 7A): several grids' runs
started together.

Counts and histograms are summed from the runs' own reports (exact: they
share their edges). Percentiles can't be merged, so the per-sample statistics
come from one query over the group's summary rows instead
(database.get_pf_datagen_group_stats). Each grid also gets a row: its size,
outputs, convergence and violation rates. Only completed runs are combined;
the others are listed with their status.
"""

from typing import Dict, List, Optional


def _sum_histograms(hists: List[dict]) -> Optional[dict]:
    hists = [h for h in hists if h]
    if not hists:
        return None
    edges = hists[0]["edges"]
    same = [h for h in hists if h["edges"] == edges]
    return {
        "edges": edges,
        "counts": [sum(h["counts"][i] for h in same) for i in range(len(edges) - 1)],
        "below": sum(h["below"] for h in same),
        "above": sum(h["above"] for h in same),
    }


def build_group_report(group: dict, runs: Dict[str, object], reports: Dict[str, dict], stats: Optional[dict],
                       per_run: Dict[str, dict]) -> dict:
    """group: the stored group (pf_datagen/run_manager.py); runs: run_id ->
    status; reports: run_id -> report of each completed run; stats / per_run:
    from database.get_pf_datagen_group_stats over the completed runs."""
    status_counts: Dict[str, int] = {}
    for r in runs.values():
        status_counts[r.status] = status_counts.get(r.status, 0) + 1
    done = [rid for rid, r in runs.items() if r.status == "completed" and rid in reports]
    outcomes = {"total": 0, "converged": 0, "failed": 0}
    reasons: Dict[str, int] = {}
    for rid in done:
        for k in outcomes:
            outcomes[k] += reports[rid].get("outcomes", {}).get(k, 0)
        for k, v in reports[rid].get("failure_reasons", {}).items():
            reasons[k] = reasons.get(k, 0) + v
    grids = []
    for entry in group["runs"]:
        rid = entry["run_id"]
        r = runs.get(rid)
        rep = reports.get(rid) or {}
        loadability = rep.get("loadability") or {}
        grids.append({
            "mv_grid_id": entry["mv_grid_id"], "run_id": rid, "status": r.status if r else "unknown",
            "n_nodes": entry.get("n_nodes"), "n_connection_points": entry.get("n_connection_points"),
            "outputs": rep.get("outcomes", {}).get("total"),
            **(per_run.get(rid) or {"converged": None, "undervoltage_rate": None, "overvoltage_rate": None,
                                    "thermal_rate": None}),
            "p_max_w": loadability.get("p_max_w"),
        })
    return {
        "group_id": group["group_id"],
        "created_at": group.get("created_at"),
        "n_runs": len(group["runs"]),
        "status_counts": status_counts,
        "complete": status_counts.get("completed", 0) == len(group["runs"]),
        "skipped": group.get("skipped", []),
        "outcomes": outcomes,
        "failure_reasons": reasons,
        "voltage_pu": _sum_histograms([reports[r].get("voltage_pu") for r in done]),
        "loading_pct": _sum_histograms([reports[r].get("loading_pct") for r in done]),
        "per_sample": stats,
        "grids": grids,
    }
