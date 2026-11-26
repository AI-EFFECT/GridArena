"""Lightweight entry points submitted to the worker pool.

The job modules import torch / diffusers / opacus. The API submits these thin
wrappers so the heavy imports only happen in the worker process, never in the
API's event loop.
"""


def run_training_job(run_dir: str) -> dict:
    from gridarena.diffusion.training import run_diffusion_training

    return run_diffusion_training(run_dir)


def run_privacy_check_job(run_dir: str) -> dict:
    from gridarena.diffusion.privacy import run_privacy_check

    return run_privacy_check(run_dir)


def run_epoch_search_job(run_dir: str) -> dict:
    from gridarena.diffusion.epoch_search import run_epoch_search

    return run_epoch_search(run_dir)


def generate_snapshot_job(run_dir: str, num_inference_steps: int, device: str) -> dict:
    from gridarena.diffusion.model_core import generate_snapshot

    return generate_snapshot(model_dir=run_dir, num_inference_steps=num_inference_steps, device=device)
