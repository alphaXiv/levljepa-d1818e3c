"""LeVLJEPA training entry point built on stable-pretraining.

Single-view, non-contrastive vision-language pretraining: image and text
embeddings predict each other's stop-gradient target through modality-specific
MLP predictors, with per-modality SIGReg preventing collapse.

    python main.py                       # uses configs/levljepa.yaml
    python main.py devices=8 batch_size=256

Multi-GPU is handled by Lightning -- set ``devices`` in the config instead of
launching with ``torchrun``.
"""

from functools import partial

import hydra
import stable_pretraining as spt
import timm
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
)
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torchvision.ops import MLP
from transformers import AutoTokenizer, GPT2Config, GPT2Model

from callbacks import (
    CheckpointSync,
    GradientClip,
    ImageNetEval,
    OnlineAttentiveProbe,
    TrainingMetrics,
)
from forwards import levljepa_forward
from utils.dataset import CC12MLanceDataset, CC12MLanceScanDataset
from utils.sigreg import SIGReg


def projector_hidden_channels(cfg: DictConfig, embed_dim: int) -> list[int]:
    """Build projector hidden/output dimensions from coarse or explicit config."""
    hidden_dims = cfg.get("projector_hidden_dims", None)
    if hidden_dims is None:
        depth = int(cfg.get("projector_depth", 1))
        if depth < 0:
            raise ValueError("projector_depth must be non-negative.")
        width = int(cfg.projector_width)
        hidden_dims = [width] * depth
    else:
        hidden_dims = [int(dim) for dim in hidden_dims]

    if any(dim <= 0 for dim in hidden_dims):
        raise ValueError("projector hidden dimensions must be positive.")
    return hidden_dims + [embed_dim]


def hidden_dims_from_config(
    cfg: DictConfig,
    explicit_key: str,
    width_key: str,
    depth_key: str,
) -> list[int]:
    """Resolve configurable hidden widths for an MLP head."""
    hidden_dims = cfg.get(explicit_key, None)
    if hidden_dims is None:
        depth = int(cfg.get(depth_key, 1))
        if depth < 0:
            raise ValueError(f"{depth_key} must be non-negative.")
        width = int(cfg.get(width_key, 2048))
        hidden_dims = [width] * depth
    else:
        hidden_dims = [int(dim) for dim in hidden_dims]

    if any(dim <= 0 for dim in hidden_dims):
        raise ValueError(f"{explicit_key} dimensions must be positive.")
    return hidden_dims


def dropout_from_config(cfg: DictConfig, key: str) -> float:
    dropout = float(cfg.get(key, 0.0))
    if dropout < 0.0 or dropout > 1.0:
        raise ValueError(f"{key} must be between 0 and 1, got {dropout}.")
    return dropout


def build_head(in_dim: int, hidden_dims: list[int], out_dim: int) -> nn.Module:
    """MLP with BN/GELU on hidden layers and a plain output projection."""
    layers = []
    prev_dim = in_dim
    for hidden_dim in hidden_dims:
        layers.extend(
            [
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
            ]
        )
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, out_dim))
    return nn.Sequential(*layers)


def linear_warmup_stable_decay(
    optimizer,
    total_steps,
    warmup_steps,
    decay_steps,
    start_factor=0.01,
    eta_min=0.0,
):
    """Warmup-Stable-Decay (WSD) schedule.

    Linear warmup (``start_factor*lr`` -> ``lr`` over ``warmup_steps``), a long
    constant hold at the peak ``lr``, then a short cosine decay (``lr`` ->
    ``eta_min``) over the final ``decay_steps``. Keeping LR high through the
    productive phase and only annealing briefly at the end avoids the long
    low-LR tail of a full cosine, where this objective overfits and the eval
    metric drifts down. ``eta_min`` is the (absolute) LR floor of the decay.

    Built from torch primitives via ``SequentialLR``, mirroring
    ``stable_pretraining.optim.LinearWarmupCosineAnnealing``.
    """
    warmup_steps = int(warmup_steps)
    decay_steps = int(decay_steps)
    stable_steps = total_steps - warmup_steps - decay_steps
    if stable_steps < 0:
        raise ValueError(
            f"WSD: warmup_steps ({warmup_steps}) + decay_steps ({decay_steps}) "
            f"exceed total_steps ({total_steps})."
        )
    warmup = LinearLR(optimizer, start_factor=start_factor, total_iters=warmup_steps)
    stable = ConstantLR(optimizer, factor=1.0, total_iters=stable_steps)
    decay = CosineAnnealingLR(optimizer, T_max=decay_steps, eta_min=eta_min)
    return SequentialLR(
        optimizer,
        [warmup, stable, decay],
        milestones=[warmup_steps, warmup_steps + stable_steps],
    )


def build_scheduler_config(cfg: DictConfig):
    """Select the LR schedule (cosine default, or WSD when ``lr_schedule=wsd``)."""
    if str(cfg.get("lr_schedule", "cosine")) == "wsd":
        decay_steps = int(cfg.get("wsd_decay_steps", 0)) or int(
            round(float(cfg.get("wsd_decay_frac", 0.15)) * int(cfg.total_steps))
        )
        return partial(
            linear_warmup_stable_decay,
            total_steps=int(cfg.total_steps),
            warmup_steps=int(cfg.warmup_steps),
            decay_steps=decay_steps,
            start_factor=float(cfg.get("warmup_start_factor", 0.01)),
            eta_min=float(cfg.eta_min),
        )
    return {
        "type": "LinearWarmupCosineAnnealing",
        "total_steps": cfg.total_steps,
        "peak_step": cfg.warmup_steps / cfg.total_steps,
        "end_lr": cfg.eta_min,
    }


def optim_config(cfg: DictConfig) -> dict:
    """AdamW optimizer + LR schedule config consumed by ``spt.Module``."""
    return {
        "optimizer": {
            "type": "AdamW",
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
        },
        "scheduler": build_scheduler_config(cfg),
        "interval": "step",
    }


def build_modules(cfg: DictConfig, vocab_size: int) -> dict:
    """Build the six trainable modules (encoders, pre-projections, predictors).

    Device placement, DDP wrapping and SyncBatchNorm conversion are all handled
    by the Lightning ``Trainer``, so none of that happens here.
    """
    hidden = cfg.model.hidden_size
    embed_dim = cfg.model.embed_dim

    text_encoder = GPT2Model(
        GPT2Config(
            n_embd=hidden,
            n_layer=cfg.model.num_layers,
            n_head=cfg.model.num_heads,
            n_inner=hidden * 4,
            vocab_size=vocab_size,
            attn_pdrop=0.0,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
        )
    )
    vision_encoder = timm.create_model(
        cfg.model.vit, pretrained=False, num_classes=0, dynamic_img_size=True
    )

    # Linear projection on the LayerNormed CLS token before SIGReg: SIGReg
    # needs variance that LayerNorm suppresses, this projection restores it.
    pre_proj_hidden = hidden_dims_from_config(
        cfg,
        explicit_key="pre_proj_hidden_dims",
        width_key="pre_proj_width",
        depth_key="pre_proj_depth",
    )

    def pre_proj():
        return build_head(hidden, pre_proj_hidden, embed_dim)

    proj_hidden = projector_hidden_channels(cfg, embed_dim)
    predictor_dropout = dropout_from_config(cfg, "predictor_dropout")

    def projector():
        return MLP(
            embed_dim,
            proj_hidden,
            norm_layer=nn.BatchNorm1d,
            activation_layer=nn.GELU,
            dropout=predictor_dropout,
        )

    return {
        "vision_encoder": vision_encoder,
        "text_encoder": text_encoder,
        "vision_pre_proj": pre_proj(),
        "text_pre_proj": pre_proj(),
        "projector_vision": projector(),
        "projector_text": projector(),
    }


@hydra.main(version_base=None, config_path="configs", config_name="levljepa")
def main(cfg: DictConfig):
    seed_everything(1, workers=True)

    torch.set_float32_matmul_precision(str(cfg.get("matmul_precision", "high")))
    spt_default_callbacks = {}
    if bool(cfg.get("spt_disable_checkpoint_callbacks", True)):
        spt_default_callbacks.update(
            {
                "sklearn_checkpoint": False,
                "wandb_checkpoint": False,
                "trackio_checkpoint": False,
                "swanlab_checkpoint": False,
                "hf_checkpoint": False,
            }
        )
    spt.set(
        default_callbacks=spt_default_callbacks,
        requeue_checkpoint=bool(cfg.get("spt_requeue_checkpoint", False)),
        progress_bar=str(cfg.get("spt_progress_bar", "simple")),
    )

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    max_samples = cfg.get("max_samples", None)
    max_text_length = cfg.get("max_text_length", 77)
    text_readout = cfg.get("text_readout", "eot")
    image_size = int(cfg.get("image_size", 224))
    image_augmentation = cfg.get("image_augmentation", "standard")
    image_crop_scale = cfg.get("image_crop_scale", (0.9, 1.0))
    image_crop_ratio = cfg.get("image_crop_ratio", (0.75, 4.0 / 3.0))
    image_color_jitter = cfg.get("image_color_jitter", None)
    image_color_jitter_prob = float(cfg.get("image_color_jitter_prob", 0.0))
    image_grayscale_prob = float(cfg.get("image_grayscale_prob", 0.0))
    image_gaussian_blur_prob = float(cfg.get("image_gaussian_blur_prob", 0.0))
    image_random_erasing_prob = float(cfg.get("image_random_erasing_prob", 0.0))
    lance_read_mode = str(cfg.get("lance_read_mode", "scan")).lower()
    if lance_read_mode in {"stream", "iter", "iterable"}:
        lance_read_mode = "scan"
    if lance_read_mode not in {"scan", "take"}:
        raise ValueError("lance_read_mode must be either 'scan' or 'take'.")

    dataset_common_kwargs = {
        "path": cfg.lance_path,
        "tokenizer": tokenizer,
        "max_length": max_text_length,
        "max_samples": max_samples,
        "image_column": cfg.get("lance_image_column", "image"),
        "text_column": cfg.get("lance_text_column", "text"),
        "text_readout": text_readout,
    }
    scan_kwargs = {}
    if lance_read_mode == "scan":
        scan_kwargs = {
            "scan_batch_size": cfg.get("lance_scan_batch_size", 2048),
            "shuffle_buffer_size": cfg.get("lance_shuffle_buffer_size", 8192),
            "batch_readahead": cfg.get("lance_batch_readahead", 4),
            "fragment_readahead": cfg.get("lance_fragment_readahead", 2),
            "scan_in_order": cfg.get("lance_scan_in_order", True),
            "seed": cfg.get("seed", 42),
        }

    dataset_cls = (
        CC12MLanceScanDataset if lance_read_mode == "scan" else CC12MLanceDataset
    )
    dataset = dataset_cls(
        **dataset_common_kwargs,
        size=image_size,
        image_augmentation=image_augmentation,
        crop_scale=image_crop_scale,
        crop_ratio=image_crop_ratio,
        color_jitter=image_color_jitter,
        color_jitter_prob=image_color_jitter_prob,
        grayscale_prob=image_grayscale_prob,
        gaussian_blur_prob=image_gaussian_blur_prob,
        random_erasing_prob=image_random_erasing_prob,
        **scan_kwargs,
    )

    print(f"[dataset] CC12M size: {len(dataset):,}")
    print(f"[dataset] lance_read_mode={lance_read_mode}")
    if lance_read_mode == "scan":
        lance_fragments = len(dataset._fragments_for_scan())
        try:
            device_count = int(cfg.devices)
        except (TypeError, ValueError):
            device_count = len(cfg.devices)
        planned_worker_shards = (
            int(cfg.num_nodes) * device_count * max(1, int(cfg.num_workers))
        )
        print(
            "[dataset] lance_fragments="
            f"{lance_fragments} planned_worker_shards={planned_worker_shards}"
        )
        if planned_worker_shards > lance_fragments:
            print(
                "[dataset] WARNING: planned worker shards exceed Lance fragments; "
                "lower num_workers to avoid duplicated fragment scans."
            )
        print(
            "[dataset] lance_scan_batch_size="
            f"{scan_kwargs['scan_batch_size']} "
            f"shuffle_buffer_size={scan_kwargs['shuffle_buffer_size']} "
            f"batch_readahead={scan_kwargs['batch_readahead']} "
            f"fragment_readahead={scan_kwargs['fragment_readahead']}"
        )
    print(
        "[dataset] image_augmentation="
        f"{image_augmentation} image_size={image_size}"
    )
    print(
        "[dataset] image_crop_scale="
        f"{list(image_crop_scale)} image_crop_ratio={list(image_crop_ratio)}"
    )
    print(
        "[dataset] image_color_jitter="
        f"{image_color_jitter} p={image_color_jitter_prob} "
        f"gray_p={image_grayscale_prob} blur_p={image_gaussian_blur_prob} "
        f"erase_p={image_random_erasing_prob}"
    )
    if max_samples is None:
        assert len(dataset) > 1_000_000, (
            f"CC12M dataset looks incomplete: only {len(dataset):,} samples"
        )

    loader_kwargs = dict(
        batch_size=cfg.batch_size,
        shuffle=lance_read_mode == "take",
        drop_last=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    if cfg.num_workers > 0:
        loader_kwargs["timeout"] = int(cfg.get("dataloader_timeout", 120))
        loader_kwargs["multiprocessing_context"] = "spawn"
        loader_kwargs["persistent_workers"] = True
    # Lightning injects the DistributedSampler under DDP; plain shuffle here.
    train_loader = DataLoader(dataset, **loader_kwargs)
    data = spt.data.DataModule(train=train_loader)

    module = spt.Module(
        **build_modules(cfg, tokenizer.vocab_size),
        sigreg=SIGReg(
            distributed=cfg.sigreg_distributed, num_proj=cfg.sigreg_num_proj
        ),
        forward=levljepa_forward,
        lambda_vision=cfg.lambda_vision,
        lambda_text=cfg.lambda_text,
        align_loss=cfg.get("align_loss", "mse"),
        text_readout=text_readout,
        optim=optim_config(cfg),
        hparams=OmegaConf.to_container(cfg, resolve=True),
    )
    callbacks = [
        GradientClip(cfg.max_grad_norm),
        LearningRateMonitor(logging_interval="step"),
        ImageNetEval(
            tokenizer=tokenizer,
            cache_dir=cfg.cache_dir,
            every_n_steps=cfg.eval_every_n_steps,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            text_readout=text_readout,
            max_text_length=max_text_length,
        ),
        TrainingMetrics(every_n_steps=cfg.get("metrics_every_n_steps", 50)),
        CheckpointSync(
            output_dir=cfg.output_dir,
            run_name=cfg.run_name,
            hf_bucket=cfg.hf_bucket,
            every_n_steps=cfg.save_every_n_steps,
        ),
    ]
    full_checkpoint_every = int(cfg.get("full_checkpoint_every_n_steps", 0))
    if full_checkpoint_every > 0:
        callbacks.append(
            ModelCheckpoint(
                filename="last",
                every_n_train_steps=full_checkpoint_every,
                every_n_epochs=0,
                save_top_k=-1,
                save_last=False,
                enable_version_counter=False,
            )
        )

    if cfg.get("online_attentive_probe", True):
        callbacks.append(
            OnlineAttentiveProbe(
                cache_dir=cfg.cache_dir,
                embed_dim=cfg.model.hidden_size,
                num_heads=cfg.model.num_heads,
                eval_every_n_steps=cfg.eval_every_n_steps,
                total_steps=cfg.total_steps,
                batch_size=cfg.get("probe_batch_size", cfg.batch_size),
                num_workers=cfg.num_workers,
                num_classes=cfg.get("probe_num_classes", 1000),
                depth=cfg.get("probe_depth", 1),
                lr=cfg.get("probe_lr", 1e-3),
                weight_decay=cfg.get("probe_weight_decay", 1e-4),
                probe_every_n_steps=cfg.get("probe_every_n_steps", 1),
            )
        )

    logger = (
        WandbLogger(project=cfg.wandb_project, name=cfg.run_name)
        if cfg.wandb_enabled
        else CSVLogger(save_dir=cfg.output_dir, name=cfg.run_name)
    )

    trainer = Trainer(
        max_steps=cfg.total_steps,
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        num_nodes=cfg.num_nodes,
        strategy="ddp" if cfg.devices != 1 else "auto",
        precision="bf16-mixed",
        sync_batchnorm=True,
        num_sanity_val_steps=0,
        limit_val_batches=0,
        log_every_n_steps=int(cfg.get("trainer_log_every_n_steps", 50)),
        enable_checkpointing=full_checkpoint_every > 0,
        use_distributed_sampler=lance_read_mode == "take",
        logger=logger,
        callbacks=callbacks,
    )

    manager_kwargs = {}
    if cfg.get("ckpt_path", None) is not None:
        manager_kwargs["ckpt_path"] = cfg.ckpt_path
        manager_kwargs["weights_only"] = bool(cfg.get("resume_weights_only", False))

    spt.Manager(trainer=trainer, module=module, data=data, **manager_kwargs)()


if __name__ == "__main__":
    main()
