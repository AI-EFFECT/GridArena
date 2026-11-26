from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field, model_validator


# Warmup never takes more than this share of the run's optimiser steps.
MAX_WARMUP_FRACTION = 0.1


def effective_warmup_steps(requested: int, total_steps: int) -> int:
    """Cap the LR warmup so short runs still reach the configured learning rate.
    A fixed 500-step warmup on a run of, say, 75 steps never gets past 15% of it."""
    return max(0, min(int(requested), int(total_steps * MAX_WARMUP_FRACTION)))


# Selectable diffusion backbones. The three custom architectures never
# downsample the node axis, so they accept any node count; "unet2d" is the
# legacy diffusers UNet2DModel (needs an even, /8-divisible node count).
Architecture = Literal["axial", "deepsets", "gnn", "unet2d"]


@dataclass
class DiffusionTrainingConfig:
    """Configuration for the diffusion model training process."""

    image_size: Tuple[int, int] = (56, 48)
    train_batch_size: int = 16
    eval_batch_size: int = 16
    num_epochs: int = 75
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 500
    save_model_epochs: int = 30
    mixed_precision: str = "no"
    seed: int = 0
    num_channels: int = 1
    power_subtract: float = 5.62
    power_rescale: float = 18.46

    # Node-mixing backbone (see Architecture above).
    architecture: str = "axial"

    # Differential privacy
    dp_enabled: bool = False
    dp_max_grad_norm: float = 1.0
    dp_noise_multiplier: float = 1.0
    dp_target_delta: float = 1e-5

    # Set by build_training_pipeline: the warmup actually used after capping.
    effective_lr_warmup_steps: Optional[int] = None


class DiffusionTrainRequest(BaseModel):
    grid_ids: List[str] = Field(..., min_length=1, description="Grid IDs to pull power data from")
    phase: Optional[Literal["R", "S", "T"]] = Field(None, description="Phase filter (omit for all)")
    start: Optional[str] = Field(None, description="Start datetime ISO (inclusive)")
    end: Optional[str] = Field(None, description="End datetime ISO (exclusive)")
    p_min: Optional[float] = Field(None, description="Min power_active (W, as stored). Days with any value below it are excluded.")
    p_max: Optional[float] = Field(None, description="Max power_active (W, as stored). Days with any value above it are excluded.")

    architecture: Architecture = Field(
        default="axial",
        description="Node-mixing backbone: 'axial' (attention, recommended), "
                    "'deepsets' (pooling, cheapest), 'gnn' (graph message passing), "
                    "or 'unet2d' (legacy 2D conv; needs even node count).",
    )

    num_epochs: int = Field(default=75, ge=1, le=1000)
    train_batch_size: int = Field(default=16, ge=1, le=256)
    learning_rate: float = Field(default=1e-4, gt=0)
    lr_warmup_steps: int = Field(default=500, ge=0)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    mixed_precision: Literal["no", "fp16", "bf16"] = "no"
    power_subtract: Union[Literal["auto"], float] = Field(
        default="auto",
        description="'auto' computes the mean from the training data, or provide a fixed float value.",
    )
    power_rescale: Union[Literal["auto"], float] = Field(
        default="auto",
        description="'auto' computes the std from the training data, or provide a fixed float value.",
    )
    device: Literal["cpu", "cuda"] = "cpu"
    seed: int = Field(default=0, ge=0, description="Seed for weight init, noise draws and batching")

    # Differential privacy
    dp_enabled: bool = Field(default=False, description="Enable DP-SGD training")
    dp_max_grad_norm: float = Field(default=1.0, gt=0, description="Per-sample gradient clip norm")
    dp_noise_multiplier: float = Field(default=1.0, gt=0, description="Noise multiplier (higher = more private)")
    dp_target_delta: float = Field(default=1e-5, gt=0, lt=1, description="Target delta for (epsilon, delta)-DP")


class DiffusionTrainResponse(BaseModel):
    run_id: str
    status: str
    message: str


class DiffusionRunStatus(BaseModel):
    run_id: str
    grid_ids: List[str]
    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    created_at: str
    completed_at: Optional[str] = None
    num_epochs: int = 0
    error: Optional[str] = None
    metrics: Optional[dict] = None
    dataset_size: int = 0
    # Which kind of job this run represents.
    job_type: Literal["train", "privacy_check", "epoch_search"] = "train"
    architecture: Optional[str] = None
    cancel_requested: bool = False


class GenerateSnapshotRequest(BaseModel):
    num_inference_steps: int = Field(default=1000, ge=10, le=2000)
    device: Literal["cpu", "cuda"] = "cpu"


class DiffusionPrivacyCheckRequest(DiffusionTrainRequest):
    """Empirical privacy audit: trains a model (on a held-out split with synthetic
    canaries injected) and measures membership-inference AUC + canary
    memorisation. This audited model is diagnostic only and is NOT shippable."""

    val_frac: float = Field(default=0.2, gt=0, lt=1, description="Fraction of days held out as non-members.")
    buffer_days: int = Field(default=7, ge=0, description="Days discarded between train and held-out split.")
    num_canaries: int = Field(default=5, ge=1, le=50, description="Synthetic canary days injected into training.")
    mi_num_timestep_samples: int = Field(default=20, ge=1, le=200, description="Timestep grid size for the loss-threshold attack.")
    num_generated_samples: int = Field(default=128, ge=8, le=1024, description="Samples generated for the canary nearest-neighbour test.")
    canary_inference_steps: int = Field(default=250, ge=10, le=2000, description="DDPM inference steps when generating the canary-test batch.")


class DiffusionEpochSearchRequest(DiffusionTrainRequest):
    """Epoch-selection sweep: trains once up to ``max_epochs``, scoring the model
    against a held-out split at a grid of checkpoint epochs, and reports the
    checkpoint with the best composite validation score."""

    max_epochs: int = Field(default=75, ge=2, le=1000, description="Maximum epochs to train during the sweep.")
    checkpoint_interval: int = Field(default=10, ge=1, le=500, description="Score the model every N epochs.")
    val_frac: float = Field(default=0.2, gt=0, lt=1, description="Fraction of days held out for validation.")
    buffer_days: int = Field(default=7, ge=0, description="Days discarded between train and validation split.")
    sweep_inference_steps: int = Field(default=100, ge=10, le=2000, description="DDPM steps used for the per-checkpoint scoring passes (fast, relative).")

    @model_validator(mode="after")
    def _no_dp(self):
        # The sweep trains without DP-SGD; accepting dp_enabled would silently
        # ship a non-private model as the run's model.
        if self.dp_enabled:
            raise ValueError("Epoch search does not support differential privacy; use a normal training run.")
        return self
