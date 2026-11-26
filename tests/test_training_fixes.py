"""Regression tests for the RL / diffusion training fixes.

Everything here runs without torch or stable-baselines3 (numpy, gymnasium and
pydantic only), so it never starts a training run. Tests that need SB3 live in
test_training_fixes_sb3.py.
"""

import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

MOCK_GRID = Path(__file__).resolve().parent.parent / "tutorials" / "mock_grids" / "lv_grid_example.json"


def _demo_grid_config(**overrides):
    from gridarena.rl.schemas import GridConfig

    grid = json.loads(MOCK_GRID.read_text(encoding="utf-8"))
    order = ["PT"] + sorted((n["node_id"] for n in grid["nodes"] if n["node_id"] != "PT"),
                            key=lambda n: int(n[1:]))
    idx = {n: i for i, n in enumerate(order)}
    cables = {c["cable_id"]: c for c in grid["cables"]}
    topology = np.array([[idx[c["from_node_id"]], idx[c["to_node_id"]]] for c in grid["connections"]])
    admittances = np.array([
        1 / ((cables[c["cable_id"]]["r_imp_real"] + 1j * cables[c["cable_id"]]["r_imp_imag"]) * c["length"] / 1000)
        for c in grid["connections"]
    ])
    return GridConfig(grid_topology=topology, admittances=admittances, n_nodes=len(order), **overrides)


# ── RL environment ──────────────────────────────────────────────────────────


def test_env_power_flow_uses_watts():
    """30 kW loads on an LV feeder must move voltages by percent, not by ppm."""
    from gridarena.rl.envs.single_agent import GridStabilityEnv

    env = GridStabilityEnv(_demo_grid_config())
    env.reset(seed=0)
    env.p_kw_state = np.full(env.n_nodes, 30.0, dtype=np.float32)
    env.p_kw_state[0] = 0.0
    env._run_power_flow()
    v_pu = np.abs(env.v_state) / env.volt_ref
    assert v_pu.min() < 0.95


def test_env_step_reports_violations_of_its_own_state():
    from gridarena.rl.envs.single_agent import GridStabilityEnv

    env = GridStabilityEnv(_demo_grid_config())
    env.reset(seed=0)
    env.p_kw_state = np.full(env.n_nodes, 30.0, dtype=np.float32)
    env.p_kw_state[0] = 0.0
    _, _, _, _, info = env.step(np.zeros(env.n_nodes, dtype=np.float32))
    v_pu = np.abs(env.v_state) / env.volt_ref
    assert info["num_violations"] == int(np.sum(np.abs(v_pu - 1.0) > env.violation_threshold))
    assert info["num_violations"] > 0


def test_env_reset_is_seeded():
    from gridarena.rl.envs.single_agent import GridStabilityEnv

    a, b = GridStabilityEnv(_demo_grid_config()), GridStabilityEnv(_demo_grid_config())
    a.reset(seed=42)
    b.reset(seed=42)
    np.testing.assert_array_equal(a.p_kw_state, b.p_kw_state)


# ── RL wrappers ─────────────────────────────────────────────────────────────


def test_stable_penalty_adds_no_l2_penalty_on_unstable_grid():
    from gridarena.rl.envs.single_agent import GridStabilityEnv
    from gridarena.rl.wrappers import StablePenalty

    base = GridStabilityEnv(_demo_grid_config())
    env = StablePenalty(GridStabilityEnv(_demo_grid_config()), unnecessary_act_weight=1.0)
    for e in (base, env.unwrapped):
        e.reset(seed=0)
        e.p_kw_state = np.full(e.n_nodes, 30.0, dtype=np.float32)
        e.p_kw_state[0] = 0.0
        e._run_power_flow()
    action = -np.ones(base.n_nodes, dtype=np.float32)
    _, r_base, *_ = base.step(action)
    _, r_wrapped, _, _, info = env.step(action)
    assert info["unnecessary_penalty"] == 0.0
    assert r_wrapped == pytest.approx(r_base)


def test_power_balance_penalty_is_in_action_units():
    from gridarena.rl.envs.single_agent import GridStabilityEnv
    from gridarena.rl.wrappers import PowerBalanceWrapper

    env = PowerBalanceWrapper(GridStabilityEnv(_demo_grid_config()), penalty_weight=1.0)
    raw = env.unwrapped
    raw.reset(seed=0)
    raw.p_kw_state = np.zeros(raw.n_nodes, dtype=np.float32)
    raw.p_kw_state[1] = 25.0  # one node, curtailed fully by action -1
    raw._run_power_flow()
    action = np.zeros(raw.n_nodes, dtype=np.float32)
    action[1] = -1.0
    _, reward, *_ = env.step(action)
    # |dP| = 25 kW = one action unit ((30 - -20) / 2): penalty 1.0, voltages fine.
    assert reward == pytest.approx(-1.0)


def test_observation_noise_is_seeded_and_does_not_mutate():
    from gridarena.rl.envs.single_agent import GridStabilityEnv
    from gridarena.rl.wrappers import ObservationNoiseWrapper

    def noisy_obs(seed):
        env = ObservationNoiseWrapper(GridStabilityEnv(_demo_grid_config()), noise_level=0.01)
        obs, _ = env.reset(seed=seed)
        return obs, env

    a, env = noisy_obs(7)
    b, _ = noisy_obs(7)
    np.testing.assert_array_equal(a["voltages_pu"], b["voltages_pu"])
    clean = env.unwrapped._get_obs()["voltages_pu"]
    assert not np.array_equal(clean, a["voltages_pu"])


def test_train_request_rejects_inverted_power_range():
    from gridarena.rl.schemas import TrainRequest

    with pytest.raises(ValidationError):
        TrainRequest(grid_id="g", p_min_kw=10.0, p_max_kw=5.0)


# ── Worker-side imports ─────────────────────────────────────────────────────


@pytest.mark.parametrize("module", ["gridarena.rl.jobs", "gridarena.diffusion.jobs",
                                    "gridarena.rl.run_manager", "gridarena.diffusion.run_manager",
                                    "gridarena.routers.rl", "gridarena.routers.diffusion"])
def test_job_entry_modules_do_not_import_ml_stack(module):
    code = (f"import sys, {module}; "
            "heavy = [m for m in ('torch', 'stable_baselines3', 'tensorflow', 'diffusers') if m in sys.modules]; "
            "print(','.join(heavy))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ""


# ── Diffusion ───────────────────────────────────────────────────────────────


def test_warmup_is_capped_to_a_fraction_of_the_run():
    from gridarena.diffusion.schemas import effective_warmup_steps

    assert effective_warmup_steps(500, 75) == 7
    assert effective_warmup_steps(500, 100_000) == 500
    assert effective_warmup_steps(0, 50) == 0


def test_epoch_search_rejects_dp():
    from gridarena.diffusion.schemas import DiffusionEpochSearchRequest

    with pytest.raises(ValidationError):
        DiffusionEpochSearchRequest(grid_ids=["g"], dp_enabled=True)


def _rows(grids=("A",), nodes=("N1", "N2"), phases=("R",), days=3, per_day=4, start=datetime(2025, 6, 2)):
    rows = []
    for d in range(days):
        for k in range(per_day):
            ts = start + timedelta(days=d, hours=6 * k)
            for gi, g in enumerate(grids):
                for ni, n in enumerate(nodes):
                    for pi, ph in enumerate(phases):
                        rows.append((g, n, ph, ts, 1.0 + 100 * gi + 10 * ni + pi + 0.01 * k))
    return rows


def test_snapshots_keep_grids_with_shared_node_names_apart():
    from gridarena.diffusion.data_loader import build_daily_snapshots

    snaps, n_nodes, per_day = build_daily_snapshots(_rows(grids=("A", "B")), require_even_nodes=False)
    assert snaps.shape == (3, 4, 4)
    assert n_nodes == 4 and per_day == 4
    assert snaps[0, :, 0].tolist() == [1.0, 11.0, 101.0, 111.0]


def test_snapshots_keep_every_phase():
    from gridarena.diffusion.data_loader import build_daily_snapshots

    snaps, n_nodes, _ = build_daily_snapshots(_rows(phases=("R", "S", "T")), require_even_nodes=False)
    assert n_nodes == 6
    assert sorted(snaps[0, :, 0].tolist()) == [1.0, 2.0, 3.0, 11.0, 12.0, 13.0]


def test_snapshots_drop_partial_and_gappy_days_instead_of_zero_filling():
    from gridarena.diffusion.data_loader import build_daily_snapshots

    rows = _rows(days=4, start=datetime(2025, 6, 2))
    rows = [r for r in rows if not (r[3].day == 2 and r[3].hour < 12)]       # partial first day
    rows = [r for r in rows if not (r[3].day == 4 and r[1] == "N2" and r[3].hour == 6)]  # gap
    snaps, _, per_day = build_daily_snapshots(rows, require_even_nodes=False)
    assert per_day == 4
    assert snaps.shape[0] == 2  # 3 Jun and 5 Jun survive
    assert not np.any(snaps == 0.0)


def test_power_range_excludes_whole_days():
    from gridarena.diffusion.data_loader import build_daily_snapshots

    rows = _rows(days=3)
    rows = [(g, n, p, ts, 99.0 if (ts.day == 3 and n == "N1" and ts.hour == 0) else v)
            for g, n, p, ts, v in rows]
    snaps, _, _ = build_daily_snapshots(rows, p_max=50.0, require_even_nodes=False)
    assert snaps.shape[0] == 2
    assert snaps.max() < 50.0


def test_normalisation_uses_the_data_it_is_given():
    from gridarena.diffusion.job_common import resolve_normalisation

    data = np.array([1.0, 3.0])
    assert resolve_normalisation({}, data) == (2.0, 1.0)
    assert resolve_normalisation({"power_subtract": 5.0, "power_rescale": 2.0}, data) == (5.0, 2.0)


def test_privacy_verdict_is_inconclusive_for_untrained_models():
    from gridarena.diffusion.privacy import audit_verdict

    assert audit_verdict(True, True, 0.98) == "inconclusive"
    assert audit_verdict(True, True, None) == "inconclusive"
    assert audit_verdict(True, True, 0.2) == "passed"
    assert audit_verdict(False, True, 0.2) == "failed"


def test_epoch_search_scores_winner_on_unseen_days():
    from gridarena.diffusion.epoch_search import split_selection_test

    p_val = np.arange(10).reshape(5, 1, 2)
    select, test = split_selection_test(p_val)
    assert len(select) == 2 and len(test) == 3
    assert not set(select[:, 0, 0]) & set(test[:, 0, 0])
