"""End-to-end walkthrough of the GridArena API.

Runs against a live GridArena server and demonstrates, in order:

  1. Grids         - upload two synthetic LV grids and inspect one of them.
  2. Measurements  - generate and upload synthetic smart-meter data, then read it back.
  3. Phase mapping - full phase-identification benchmark pipeline
                     (training data -> test data -> submit guesses -> score).
  4. RL training   - start a voltage-control PPO run, watch it start, then stop it.
  5. Diffusion     - start a diffusion-model training run, watch it start, then stop it.
  6. Digital twin  - create a twin fed by the Sentinel PV API and run one cycle.

All grid and measurement data is synthetic: the topology comes from
tutorials/mock_grids/lv_grid_example.json and the measurements are generated
here. The digital twin reads live PV data from the Sentinel API, which needs the
SENTINEL_AUTH_TOKEN environment variable set *on the server* (e.g. in its .env),
and SENTINEL_URL set where this script runs; without SENTINEL_URL that step is skipped.

Usage:
    python tutorials/demo/gridarena_demo.py
    python tutorials/demo/gridarena_demo.py --base-url http://localhost:8001
    python tutorials/demo/gridarena_demo.py --skip-training --cleanup
    python tutorials/demo/gridarena_demo.py --keep-training   # let the runs finish

The demo objects are kept afterwards so they can be browsed in the UI; re-running
the script replaces them. Pass --cleanup to delete them at the end.
"""

import argparse
import cmath
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

MOCK_GRID = Path(__file__).resolve().parent.parent / "mock_grids" / "lv_grid_example.json"

TRAIN_GRID = "DEMO_LV_TRAIN"
TEST_GRID = "DEMO_LV_TEST"
TWIN_NAME = "GridArena demo twin"
GUESS_ID = "demo_angle_baseline"

PHASES = ("R", "S", "T")
PHASE_ANGLE_DEG = {"R": 0.0, "S": -120.0, "T": 120.0}
SLOTS_PER_DAY = 48  # 30-minute resolution

# URL of the Sentinel PV data API used by the digital-twin step. Not public:
# set SENTINEL_URL in the environment, e.g.
#   SENTINEL_URL=https://sentinel.example.com/data/observed/pv/
SENTINEL_URL = os.environ.get("SENTINEL_URL", "")


# ── Output helpers ───────────────────────────────────────────────────────


def section(title):
    print(f"\n{'=' * 72}\n  {title}\n{'=' * 72}")


def step(text):
    print(f"\n-> {text}")


def show(label, value):
    print(f"   {label:<28} {value}")


class Api:
    """Thin wrapper around httpx that fails loudly on unexpected status codes."""

    def __init__(self, base_url):
        self.base = base_url.rstrip("/")
        self.client = httpx.Client(base_url=self.base, timeout=300)

    def call(self, method, path, expect=(200,), **kwargs):
        resp = self.client.request(method, path, **kwargs)
        if resp.status_code not in expect:
            raise RuntimeError(f"{method} {path} -> {resp.status_code}: {resp.text[:500]}")
        return resp

    def get(self, path, **kwargs):
        return self.call("GET", path, **kwargs).json()

    def post(self, path, **kwargs):
        return self.call("POST", path, **kwargs).json()

    def ui(self, path):
        return f"{self.base}{path}"


# ── Synthetic data ───────────────────────────────────────────────────────


def build_tree(connections):
    children = defaultdict(list)
    for c in connections:
        children[c["from_node_id"]].append(c["to_node_id"])
    return children


def print_tree(children, node="PT", prefix=""):
    kids = children.get(node, [])
    for i, child in enumerate(kids):
        last = i == len(kids) - 1
        print(f"   {prefix}{'└─' if last else '├─'} {child}")
        print_tree(children, child, prefix + ("   " if last else "│  "))


def node_depths(children, root="PT"):
    depths, stack = {root: 0}, [root]
    while stack:
        node = stack.pop()
        for child in children.get(node, []):
            depths[child] = depths[node] + 1
            stack.append(child)
    return depths


def assign_phases(node_ids, seed):
    """Single-phase customers spread over R/S/T (deterministic per grid)."""
    rng = random.Random(seed)
    return {n: rng.choice(PHASES) for n in node_ids if n != "PT"}


def load_kw(slot, rng, has_pv):
    """Residential profile: morning and evening peaks, optional midday PV."""
    hour = slot / 2
    base = 0.4 + 0.9 * math.exp(-((hour - 8) ** 2) / 3) + 1.6 * math.exp(-((hour - 20) ** 2) / 4)
    p = base * rng.uniform(0.8, 1.2)
    if has_pv:
        p -= max(0.0, 2.5 * math.sin(math.pi * (hour - 6) / 13)) if 6 <= hour <= 19 else 0.0
    return round(p, 3)


def generate_measurements(grid, phase_of, days, seed):
    """Synthetic per-phase measurements for every node of the grid.

    Each phase has its own slowly varying voltage at the transformer; customers see
    that voltage minus a drop that grows with their load and distance from the PT.
    Voltage angles sit near the phase's nominal angle (0, -120, +120 degrees).
    """
    rng = random.Random(seed)
    depths = node_depths(build_tree(grid["connections"]))
    has_pv = {n: rng.random() < 0.4 for n in phase_of}
    start = datetime(2025, 6, 2)

    series = defaultdict(list)  # (node_id, phase) -> list of records
    v_walk = {ph: 0.0 for ph in PHASES}
    for slot_idx in range(days * SLOTS_PER_DAY):
        ts = (start + timedelta(minutes=30 * slot_idx)).isoformat()
        slot = slot_idx % SLOTS_PER_DAY
        for ph in PHASES:
            v_walk[ph] = 0.95 * v_walk[ph] + rng.gauss(0, 0.4)
        v_pt = {ph: 235.0 + v_walk[ph] for ph in PHASES}

        for ph in PHASES:
            series[("PT", ph)].append(_record(ts, ph, 0.0, 0.0, v_pt[ph], PHASE_ANGLE_DEG[ph]))

        for node, ph in phase_of.items():
            p = load_kw(slot, rng, has_pv[node])
            q = round(0.3 * max(p, 0.0), 3)
            v = v_pt[ph] - 0.8 * depths.get(node, 1) * p + rng.gauss(0, 0.15)
            ang = PHASE_ANGLE_DEG[ph] - 0.4 * depths.get(node, 1) * p + rng.gauss(0, 0.2)
            series[(node, ph)].append(_record(ts, ph, p, q, v, ang))

    by_node = defaultdict(list)
    for (node, _ph), recs in series.items():
        by_node[node].extend(recs)
    return {
        "grid_id": grid["grid_id"],
        "historical": [{"node_id": n, "measurements": recs} for n, recs in by_node.items()],
    }


def _record(ts, phase, p, q, v, ang):
    return {
        "datetime": ts,
        "phase": phase,
        "power_active": p,
        "power_reactive": q,
        "voltage_magnitude": round(v, 3),
        "voltage_angle": round(ang, 3),
    }


def circular_mean_deg(angles):
    z = sum(cmath.exp(1j * math.radians(a)) for a in angles) / len(angles)
    return math.degrees(cmath.phase(z))


def angle_gap(a, b):
    return abs((a - b + 180) % 360 - 180)


# ── Demo steps ───────────────────────────────────────────────────────────


def reset_demo_objects(api):
    """Remove objects left by a previous run so the demo is repeatable."""
    for twin in api.get("/digital-twins/"):
        if twin.get("name") == TWIN_NAME:
            api.call("DELETE", f"/digital-twins/{twin['digital_twin_id']}")
    for grid_id in (TRAIN_GRID, TEST_GRID):
        api.call("DELETE", f"/grid/{grid_id}", expect=(200, 404))


def demo_grids(api):
    section("1. Grids")
    base = json.loads(MOCK_GRID.read_text(encoding="utf-8"))
    grids = {}
    for grid_id, role in ((TRAIN_GRID, "train"), (TEST_GRID, "test")):
        grid = dict(base, grid_id=grid_id)
        flags = {k: "NO" for k in (
            "phase_detection_test", "phase_detection_train",
            "topology_detection_test", "topology_detection_train",
            "voltage_control_test", "voltage_control_train",
            "state_estimation_test", "state_estimation_train",
        )}
        flags[f"phase_detection_{role}"] = "YES"

        step(f"Validating and uploading {grid_id} (phase detection: {role})")
        files = {"file": (f"{grid_id}.json", json.dumps(grid), "application/json")}
        check = api.post("/grid/validate", files=files)
        show("valid", f"{check['valid']}  ({len(grid['nodes'])} nodes, "
                      f"{len(grid['cables'])} cable types, {len(grid['connections'])} connections)")
        files = {"file": (f"{grid_id}.json", json.dumps(grid), "application/json")}
        show("upload", api.post("/grid/", params=flags, files=files)["message"])
        grids[grid_id] = grid

    step(f"Reading {TEST_GRID} back from the database")
    nodes = api.get(f"/grid/data/{TEST_GRID}", params={"table": "Node"})
    cables = api.get(f"/grid/data/{TEST_GRID}", params={"table": "Cable"})
    conns = api.get(f"/grid/data/{TEST_GRID}", params={"table": "Connection"})
    show("nodes", _count(nodes))
    show("cable types", _count(cables))
    show("connections", _count(conns))
    print("\n   Topology (radial, from the transformer point):\n   PT")
    print_tree(build_tree(grids[TEST_GRID]["connections"]))
    print()
    show("UI", api.ui(f"/grid/ui/{TEST_GRID}"))
    return grids


def _count(payload):
    if isinstance(payload, dict):
        for key in ("data", "rows", "records"):
            if isinstance(payload.get(key), list):
                return len(payload[key])
    return len(payload) if isinstance(payload, list) else payload


def demo_measurements(api, grids, days):
    section("2. Measurements")
    for i, (grid_id, grid) in enumerate(grids.items()):
        phase_of = assign_phases([n["node_id"] for n in grid["nodes"]], seed=f"{grid_id}-phases")
        data = generate_measurements(grid, phase_of, days, seed=1000 + i)
        n_records = sum(len(n["measurements"]) for n in data["historical"])
        step(f"Uploading {n_records} synthetic records for {grid_id} ({days} days, 30-min resolution)")
        t0 = time.time()
        files = {"file": (f"{grid_id}_measurements.json", json.dumps(data), "application/json")}
        show("upload", api.post("/measurements/", files=files)["message"])
        show("took", f"{time.time() - t0:.1f} s")

    step(f"Reading node N3 of {TEST_GRID} for the first day")
    first_day = datetime(2025, 6, 2)
    resp = api.get(
        f"/measurements/data/{TEST_GRID}",
        params={"node_id": "N3", "start": first_day.isoformat(),
                "end": (first_day + timedelta(days=1)).isoformat()},
    )
    rows = resp["records"]
    print(f"\n   {'datetime':<20} {'ph':<3} {'P (W)':>8} {'Q (VAr)':>9} {'|V| (V)':>9} {'angle':>8}")
    for r in rows[16:24]:  # 08:00-12:00
        print(f"   {str(r['datetime'])[:19]:<20} {r['phase']:<3} {r['power_active']:>8.3f} "
              f"{r['power_reactive']:>9.3f} {r['voltage_magnitude']:>9.2f} {r['voltage_angle']:>8.2f}")
    volts = [r["voltage_magnitude"] for r in rows]
    print()
    show("rows that day", len(rows))
    show("voltage range", f"{min(volts):.1f} - {max(volts):.1f} V")
    show("UI", api.ui(f"/measurements/ui/{TEST_GRID}"))


def demo_phase_mapping(api, difficulty):
    section("3. Phase identification benchmark")

    step(f"Downloading labelled training data (difficulty={difficulty})")
    train = api.get("/phase_identification_benchmark/training-phase-data",
                    params={"difficulty": difficulty})
    show("noise profile", train["profile"])
    angles = defaultdict(list)
    for grid_id, nodes in train["grids"].items():
        for records in nodes.values():
            for r in records:
                angles[r["phase"]].append(r["voltage_angle"])
    centroids = {ph: circular_mean_deg(a) for ph, a in angles.items()}
    show("training grids", ", ".join(train["grids"]))
    for ph in PHASES:
        show(f"mean voltage angle {ph}", f"{centroids[ph]:7.2f} deg  ({len(angles[ph])} samples)")

    step("Downloading anonymised test data (phase labels removed)")
    test = api.get("/phase_identification_benchmark/phase-data", params={"difficulty": difficulty})
    series = test["grids"].get(TEST_GRID, {})
    show("anonymised series", len(series))
    show("example key", next(iter(series), "-"))

    step("Classifying each series by its mean voltage angle (nearest training centroid)")
    guesses = {}
    for key, records in series.items():
        mean_angle = circular_mean_deg([r["voltage_angle"] for r in records])
        guesses[key] = min(centroids, key=lambda ph: angle_gap(mean_angle, centroids[ph]))
    counts = {ph: list(guesses.values()).count(ph) for ph in PHASES}
    show("guesses per phase", counts)

    step(f"Submitting guesses as '{GUESS_ID}'")
    show("response", api.post("/phase_identification_benchmark/submit-results",
                              json={"guess_id": GUESS_ID, "grid_id": TEST_GRID, "guesses": guesses}))

    step("Retrieving the score (computed against the hidden ground truth)")
    score = api.get("/phase_identification_benchmark/score",
                    params={"guess_id": GUESS_ID, "grid_id": TEST_GRID})
    show("accuracy", f"{score['accuracy']:.1%}")
    show("UI", api.ui("/phase_identification_benchmark/ui"))


def watch_run(api, path, seconds):
    """Poll a background run briefly so the status transition is visible."""
    deadline, status = time.time() + seconds, None
    while time.time() < deadline:
        run = api.get(path)
        if run.get("status") != status:
            status = run.get("status")
            show(f"status @ {time.strftime('%H:%M:%S')}", status)
        if status in ("completed", "failed", "cancelled"):
            break
        time.sleep(2)
    return status


def stop_or_leave_running(api, kind, run_id, keep):
    """Cancel the run (default) so it doesn't keep a slow machine busy."""
    base = f"/{kind}/runs/{run_id}"
    if keep:
        print("   Not waiting for completion - training continues in the background.")
        show("follow progress", api.ui(f"/{kind}/ui/{run_id}"))
        show("stop it with", f"POST {base}/cancel")
        return
    step("Stopping the run (pass --keep-training to let it finish)")
    resp = api.call("POST", f"{base}/cancel", expect=(200, 409)).json()
    show("cancel", resp.get("message") or resp.get("detail"))
    watch_run(api, base, 60)
    show("run page", api.ui(f"/{kind}/ui/{run_id}"))


def demo_rl(api, timesteps, watch, keep):
    section("4. RL training (voltage control)")
    request = {"grid_id": TEST_GRID, "agent_type": "single", "timesteps": timesteps, "device": "cpu"}
    step("Starting a PPO training run")
    print("   POST /rl/train " + json.dumps(request))
    resp = api.post("/rl/train", json=request)
    show("run_id", resp["run_id"])
    show("message", resp["message"])
    watch_run(api, f"/rl/runs/{resp['run_id']}", watch)
    stop_or_leave_running(api, "rl", resp["run_id"], keep)


def demo_diffusion(api, epochs, watch, keep):
    section("5. Diffusion model training (synthetic data generation)")
    request = {"grid_ids": [TRAIN_GRID], "architecture": "axial", "num_epochs": epochs,
               "device": "cpu", "dp_enabled": False}
    step("Starting a diffusion training run")
    print("   POST /diffusion/train " + json.dumps(request))
    resp = api.post("/diffusion/train", json=request)
    show("run_id", resp["run_id"])
    show("message", resp["message"])
    watch_run(api, f"/diffusion/runs/{resp['run_id']}", watch)
    stop_or_leave_running(api, "diffusion", resp["run_id"], keep)


def demo_digital_twin(api):
    section("6. Digital twin (live Sentinel PV data)")
    if not SENTINEL_URL:
        print("SENTINEL_URL is not set; skipping the digital twin step.")
        return
    now = datetime.now(timezone.utc).replace(microsecond=0)
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    # The Sentinel API requires a date window; it is fixed at creation time, so it
    # spans the next day as well to keep periodic ticks fed.
    node_params = {
        "variable": "pv_1_pac",
        "start_date": (now - timedelta(days=1)).strftime(fmt),
        "end_date": (now + timedelta(days=1)).strftime(fmt),
    }
    config = {
        "grid_id": TEST_GRID,
        "name": TWIN_NAME,
        "description": "Created by tutorials/demo/gridarena_demo.py",
        "update_interval_seconds": 300,
        "source_config": {
            "source_type": "http",
            "url": SENTINEL_URL,
            "method": "GET",
            "auth_type": "token",
            "secret_ref": "SENTINEL_AUTH_TOKEN",
            "secret_header": "Authorization",
        },
        "node_assignments": [{
            "node_id": "N1",
            "phase": "R",
            "query_params": node_params,
            "mapping": {"timestamp": "data[0].datetime", "active_power": "data[0].value"},
        }],
        "powerflow_config": {"phase": "R", "voltage_ref": 230.0},
    }

    step("Creating the digital twin")
    print("   POST /digital-twins/ " + json.dumps(config, indent=2).replace("\n", "\n   "))
    twin_id = api.post("/digital-twins/", json=config)["digital_twin_id"]
    show("digital_twin_id", twin_id)

    step("Probing the source with node N1's query parameters")
    probe = api.post(f"/digital-twins/{twin_id}/probe", json={"query_params": node_params})
    show("success", probe.get("success"))
    show("HTTP status", probe.get("status_code"))
    payload = probe.get("raw_payload") or {}
    latest = (payload.get("data") or [None])[0] if isinstance(payload, dict) else None
    if latest:
        show("latest reading (data[0])", f"{latest['value']} {latest.get('units', '')} @ {latest['datetime']}")
    elif probe.get("error"):
        show("error", probe["error"])

    step("Running one cycle: fetch -> map -> power flow -> compare")
    tick = api.call("POST", f"/digital-twins/{twin_id}/tick", expect=(200, 500)).json()
    if "metrics" in tick:
        for key, value in (tick["metrics"] or {}).items():
            if not isinstance(value, (dict, list)):
                show(key, value)
    else:
        show("tick failed", tick.get("detail"))

    state = api.get(f"/digital-twins/{twin_id}/latest-state")
    batch = state.get("latest_measurements") or {}
    for m in (batch.get("measurements") or [])[:3]:
        show(f"measurement {m['node_id']}/{m['phase']}", f"P={m['power_active']} @ {m['datetime']}")
    show("UI", api.ui(f"/digital-twins/ui/{twin_id}"))
    print("   The twin is left stopped; POST /digital-twins/{id}/start runs it periodically.")


def cleanup(api):
    section("Cleanup")
    reset_demo_objects(api)
    print("   Demo grids, measurements and digital twin deleted.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://localhost:8001")
    parser.add_argument("--days", type=int, default=7, help="days of synthetic measurements per grid")
    parser.add_argument("--difficulty", default="medium", choices=["clean", "easy", "medium", "hard"])
    parser.add_argument("--rl-timesteps", type=int, default=2000, help="PPO timesteps (min 1000)")
    parser.add_argument("--diffusion-epochs", type=int, default=5)
    parser.add_argument("--watch", type=int, default=10, help="seconds to watch each training run")
    parser.add_argument("--skip-training", action="store_true", help="skip the RL and diffusion runs")
    parser.add_argument("--keep-training", action="store_true",
                        help="leave the RL and diffusion runs going instead of stopping them")
    parser.add_argument("--skip-twin", action="store_true", help="skip the digital twin")
    parser.add_argument("--cleanup", action="store_true", help="delete the demo objects at the end")
    args = parser.parse_args()

    api = Api(args.base_url)
    try:
        info = api.get("/openapi.json")["info"]
    except Exception as e:
        sys.exit(f"Cannot reach GridArena at {args.base_url}: {e}")
    print(f"Connected to {info['title']} {info.get('version', '')} at {api.base}")

    reset_demo_objects(api)
    grids = demo_grids(api)
    demo_measurements(api, grids, args.days)
    demo_phase_mapping(api, args.difficulty)
    if not args.skip_training:
        demo_rl(api, args.rl_timesteps, args.watch, args.keep_training)
        demo_diffusion(api, args.diffusion_epochs, args.watch, args.keep_training)
    if not args.skip_twin:
        demo_digital_twin(api)
    if args.cleanup:
        cleanup(api)
    section("Done")


if __name__ == "__main__":
    main()
