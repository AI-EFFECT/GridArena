"""Lightweight entry points submitted to the worker pool.

Importing ``gridarena.rl.training`` pulls in stable-baselines3 / torch. Doing that
in the API process blocks the event loop (and has hung it on slow disks), so the
API submits these thin wrappers and the heavy import only happens in the worker.
"""


def run_training_job(run_dir: str) -> dict:
    from gridarena.rl.training import run_training

    return run_training(run_dir)


def evaluate_job(run_dir: str, loads=None) -> dict:
    from gridarena.rl.evaluation import evaluate_model

    return evaluate_model(run_dir, loads=loads)
