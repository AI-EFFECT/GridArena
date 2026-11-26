"""Regression tests for the RL fixes that need stable-baselines3 / torch.

They build environments and PPO objects but never call learn(), so no
training run starts.
"""

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")

from test_training_fixes import _demo_grid_config  # noqa: E402


def test_multi_agent_env_excludes_pt_and_identifies_nodes():
    from gridarena.rl.envs.multi_agent import MultiGridStabilityEnv, PettingZooVecEnvWrapper

    cfg = _demo_grid_config()
    vec = PettingZooVecEnvWrapper(MultiGridStabilityEnv(cfg))
    assert vec.num_envs == cfg.n_nodes - 1
    obs = vec.reset()
    assert obs["node_index"].shape == (cfg.n_nodes - 1, 1)
    assert len(np.unique(obs["node_index"])) == cfg.n_nodes - 1


def test_multi_agent_infos_carry_violations_before_reset():
    from gridarena.rl.envs.multi_agent import MultiGridStabilityEnv, PettingZooVecEnvWrapper

    env = MultiGridStabilityEnv(_demo_grid_config())
    vec = PettingZooVecEnvWrapper(env)
    vec.reset()
    env.p_kw_state = np.full(env.n_nodes, 30.0, dtype=np.float32)
    env.p_kw_state[0] = 0.0
    env._run_power_flow()
    _, _, dones, infos = vec.step(np.zeros((vec.num_envs, 1), dtype=np.float32))
    assert dones.all()
    assert infos[0]["num_violations"] > 0


def test_multi_agent_reset_is_seeded():
    from gridarena.rl.envs.multi_agent import MultiGridStabilityEnv

    a, b = MultiGridStabilityEnv(_demo_grid_config()), MultiGridStabilityEnv(_demo_grid_config())
    a.reset(seed=3)
    b.reset(seed=3)
    np.testing.assert_array_equal(a.p_kw_state, b.p_kw_state)


def test_rollout_fits_short_budgets():
    from gridarena.rl.training import _rollout_kwargs

    assert _rollout_kwargs(1000, 1) == {"n_steps": 1000, "batch_size": 64}
    assert _rollout_kwargs(50_000, 10) == {"n_steps": 2048, "batch_size": 64}
