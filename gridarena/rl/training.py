"""Top-level training functions executed in subprocesses via ProcessPoolExecutor."""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from gridarena.rl.envs.multi_agent import MultiGridStabilityEnv, PettingZooVecEnvWrapper
from gridarena.rl.envs.single_agent import GridStabilityEnv
from gridarena.rl.schemas import GridConfig, WrapperConfig
from gridarena.rl.wrappers import apply_wrappers
from gridarena.run_cancellation import CancelCheck, is_cancel_requested

logger = logging.getLogger(__name__)


class CancelOnRequest(BaseCallback):
    """Stops model.learn() once the run's cancel marker appears (returning
    False from _on_step is SB3's supported way to end training early)."""

    def __init__(self, run_dir: str, check_every: int = 64):
        super().__init__(0)
        self.check = CancelCheck(run_dir, every=check_every)

    def _on_step(self) -> bool:
        return not self.check()


class TrainingLogger(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_violations = []
        self._current_ep_reward = 0.0

    def _on_step(self) -> bool:
        self._current_ep_reward += self.locals["rewards"][0]
        if self.locals["dones"][0]:
            self.episode_rewards.append(self._current_ep_reward)
            # Read from the step info: by now the vec env has already reset, so
            # the env's own state belongs to the next episode.
            self.episode_violations.append(int(self.locals["infos"][0].get("num_violations", 0)))
            self._current_ep_reward = 0.0
        return True


class MultiAgentTrainingLogger(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_violations = []
        self._current_ep_reward = 0.0

    def _on_step(self) -> bool:
        self._current_ep_reward += self.locals["rewards"][0]
        if self.locals["dones"][0]:
            self.episode_rewards.append(self._current_ep_reward)
            # Read from the step info: by now the vec env has already reset, so
            # the env's own state belongs to the next episode.
            self.episode_violations.append(int(self.locals["infos"][0].get("num_violations", 0)))
            self._current_ep_reward = 0.0
        return True


def _rollout_kwargs(timesteps: int, num_envs: int) -> dict:
    """PPO rollout sizes that fit the requested budget. SB3's default n_steps=2048
    would make every run shorter than 2048 episodes train for 2048 anyway."""
    n_steps = max(1, min(2048, timesteps))
    return {"n_steps": n_steps, "batch_size": max(1, min(64, n_steps * num_envs))}


def _update_run_config(run_dir: str, updates: dict):
    config_path = Path(run_dir) / "run_config.json"
    with open(config_path) as f:
        data = json.load(f)
    data.update(updates)
    with open(config_path, "w") as f:
        json.dump(data, f, indent=2)


def _mark_cancelled(run_dir: str) -> dict:
    _update_run_config(run_dir, {
        "status": "cancelled",
        "completed_at": datetime.now(timezone.utc).isoformat(),
    })
    logger.info("Training in %s cancelled", run_dir)
    return {"status": "cancelled"}


def run_training(run_dir: str) -> dict:
    """Top-level picklable function for subprocess execution.

    Reads config from run_dir/run_config.json, trains the model,
    saves artifacts, and updates the config with results.
    """
    import matplotlib
    matplotlib.use("Agg")

    config_path = Path(run_dir) / "run_config.json"
    with open(config_path) as f:
        run_data = json.load(f)

    if is_cancel_requested(run_dir):
        return _mark_cancelled(run_dir)

    _update_run_config(run_dir, {"status": "running"})
    stopper = CancelOnRequest(run_dir)

    try:
        grid_config = GridConfig.from_dict(run_data["grid_config"])
        agent_type = run_data["agent_type"]
        timesteps = run_data["timesteps"]
        active_wrappers = run_data.get("active_wrappers", [])
        wrapper_config = WrapperConfig(**run_data.get("wrapper_config", {}))
        device = run_data.get("device", "cpu")
        seed = run_data.get("seed")

        if agent_type == "single":
            model_name = "ppo_grid"
            rewards_name = "rewards.npy"
            violations_name = "violations.npy"

            def make_env():
                env = GridStabilityEnv(grid_config)
                env = apply_wrappers(env, active_wrappers, wrapper_config)
                env = Monitor(env)
                return env

            vec_env = DummyVecEnv([make_env])
            model = PPO(policy="MultiInputPolicy", env=vec_env, verbose=0, device=device,
                        seed=seed, **_rollout_kwargs(timesteps, vec_env.num_envs))
            cb = TrainingLogger()
            model.learn(total_timesteps=timesteps, callback=CallbackList([cb, stopper]), progress_bar=False)

        else:
            model_name = "ppo_multiagent"
            rewards_name = "marl_rewards.npy"
            violations_name = "marl_violations.npy"

            raw_env = MultiGridStabilityEnv(grid_config)
            vec_env = PettingZooVecEnvWrapper(raw_env)
            model = PPO(policy="MultiInputPolicy", env=vec_env, verbose=0, device=device,
                        seed=seed, **_rollout_kwargs(timesteps, vec_env.num_envs))
            cb = MultiAgentTrainingLogger()
            # One vec step = one episode for every agent; "timesteps" counts episodes.
            total = timesteps * vec_env.num_envs
            model.learn(total_timesteps=total, callback=CallbackList([cb, stopper]), progress_bar=False)

        if stopper.check.cancelled:
            return _mark_cancelled(run_dir)

        model.save(os.path.join(run_dir, model_name))
        np.save(os.path.join(run_dir, rewards_name), np.array(cb.episode_rewards))
        np.save(os.path.join(run_dir, violations_name), np.array(cb.episode_violations))

        metrics = {
            "total_episodes": len(cb.episode_rewards),
            "mean_reward": float(np.mean(cb.episode_rewards[-100:])) if cb.episode_rewards else 0.0,
            "mean_violations": float(np.mean(cb.episode_violations[-100:])) if cb.episode_violations else 0.0,
        }

        _update_run_config(run_dir, {
            "status": "completed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics,
        })

        return {"status": "completed", "metrics": metrics}

    except Exception as e:
        logger.error("Training failed: %s", e, exc_info=True)
        _update_run_config(run_dir, {
            "status": "failed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "error": str(e),
        })
        return {"status": "failed", "error": str(e)}
