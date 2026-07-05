"""Lightning callbacks replicating LeVLJEPA's logging, evaluation, gradient
clipping and checkpointing on top of stable-pretraining."""

import os

import torch
import torch.nn.functional as F
import wandb
from lightning.pytorch import Callback
from torch.utils.data import DataLoader

from huggingface_hub import sync_bucket

from utils.attentive import AttentiveClassifier
from utils.eval_utils import ImageNetTrain, ImageNetVal, run_imagenet_eval


def _is_global_zero(trainer) -> bool:
    for env_name in ("SLURM_PROCID", "RANK", "GLOBAL_RANK"):
        value = os.environ.get(env_name)
        if value is not None:
            return int(value) == 0
    return trainer.is_global_zero


def _effective_rank(x: torch.Tensor) -> torch.Tensor:
    centered = x - x.mean(0)
    sv = torch.linalg.svdvals(centered.float())
    sv_norm = sv / sv.sum()
    return torch.exp(-torch.sum(sv_norm * torch.log(sv_norm + 1e-7)))


class GradientClip(Callback):
    """Clips the global gradient norm before each optimizer step.

    stable-pretraining drives optimization manually, so ``Trainer``'s own
    ``gradient_clip_val`` is unavailable; this hook reproduces the original
    ``clip_grad_norm_`` call.
    """

    def __init__(self, max_norm: float):
        super().__init__()
        self.max_norm = max_norm

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        torch.nn.utils.clip_grad_norm_(pl_module.parameters(), self.max_norm)


class TrainingMetrics(Callback):
    """Logs cross-modal alignment diagnostics from each training batch.

    Mirrors the per-step metrics block of the original train scripts:
    contrastive logit gaps, cosine similarities (matched vs. rolled negatives)
    and embedding effective rank.
    """

    def __init__(self, every_n_steps: int = 1):
        super().__init__()
        self.every = max(1, int(every_n_steps))

    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not _is_global_zero(trainer):
            return
        if trainer.global_step % self.every != 0:
            return
        if not isinstance(outputs, dict) or "image_linear" not in outputs:
            return

        image_linear = outputs["image_linear"].float()
        text_linear = outputs["text_linear"].float()
        image_proj = outputs["image_proj"].float()
        text_proj = outputs["text_proj"].float()

        # Prediction alignment: image_proj predicts the text embedding.
        v_proj = F.normalize(image_proj, dim=-1)
        t_proj = F.normalize(text_linear, dim=-1)
        logits_proj = v_proj @ t_proj.T / 0.07
        pos_logits_proj = logits_proj.diagonal().mean()
        lse_proj = torch.logsumexp(logits_proj, dim=1).mean()

        v_base = F.normalize(image_linear, dim=-1)
        t_base = F.normalize(text_linear, dim=-1)
        logits_base = v_base @ t_base.T / 0.07
        pos_logits_base = logits_base.diagonal().mean()
        lse_base = torch.logsumexp(logits_base, dim=1).mean()

        metrics = {
            "pos_logits_proj": pos_logits_proj,
            "lse_proj": lse_proj,
            "gap_proj": pos_logits_proj - lse_proj,
            "pos_logits_base": pos_logits_base,
            "lse_base": lse_base,
            "gap_base": pos_logits_base - lse_base,
            "cos_sim_matched": F.cosine_similarity(image_linear, text_linear).mean(),
            "cos_sim_random": F.cosine_similarity(
                image_linear, text_linear.roll(1, 0)
            ).mean(),
            "effective_rank_vision": _effective_rank(image_linear),
            "effective_rank_text": _effective_rank(text_linear),
            "cos_sim_img_pred": F.cosine_similarity(image_proj, text_linear).mean(),
            "cos_sim_img_pred_random": F.cosine_similarity(
                image_proj, text_linear.roll(1, 0)
            ).mean(),
            "cos_sim_txt_pred": F.cosine_similarity(text_proj, image_linear).mean(),
            "cos_sim_txt_pred_random": F.cosine_similarity(
                text_proj, image_linear.roll(1, 0)
            ).mean(),
        }
        trainer.logger.log_metrics(
            {k: float(v) for k, v in metrics.items()}, step=trainer.global_step
        )


class ImageNetEval(Callback):
    """Periodic ImageNet zero-shot (and optional linear probe) evaluation.

    Wraps ``run_imagenet_eval`` and fires every ``every_n_steps`` optimizer
    steps on rank zero, exactly as the original scripts did. ImageNet enters
    through this callback's own dataloader; it is never part of the
    pretraining ``DataModule``.
    """

    def __init__(
        self,
        tokenizer,
        cache_dir,
        every_n_steps,
        batch_size,
        num_workers=8,
        text_readout="eot",
        max_text_length=77,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.cache_dir = cache_dir
        self.every = every_n_steps
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.text_readout = text_readout
        self.max_text_length = max_text_length

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step == 0 or step % self.every != 0:
            return
        if _is_global_zero(trainer):
            try:
                metrics = run_imagenet_eval(
                    vision_model=pl_module.vision_encoder,
                    text_model=pl_module.text_encoder,
                    tokenizer=self.tokenizer,
                    cache_dir=self.cache_dir,
                    device=pl_module.device,
                    vision_embed=pl_module.vision_pre_proj,
                    text_embed=pl_module.text_pre_proj,
                    proj_vision=pl_module.projector_vision,
                    proj_text=pl_module.projector_text,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    text_readout=self.text_readout,
                    max_text_length=self.max_text_length,
                )
                print(f"[eval] step={step} metrics={metrics}", flush=True)
                if trainer.logger is not None:
                    trainer.logger.log_metrics(metrics, step=step)
                if wandb.run is not None:
                    print(
                        f"[eval] logging metrics to wandb run={wandb.run.id}",
                        flush=True,
                    )
                    wandb.log({"trainer/global_step": step, **metrics}, commit=True)
                    wandb.run.summary.update(metrics)
            except Exception as e:  # eval must never crash training
                print(f"[eval] failed with: {e}")
        trainer.strategy.barrier()


class OnlineAttentiveProbe(Callback):
    """Online attentive-probe classifier trained on ImageNet during pretraining.

    An ``AttentiveClassifier`` (cross-attention pooler + linear head) is trained
    on top of *frozen, detached* vision-encoder patch tokens, using its own
    optimizer/scheduler. Because the CC12M pretraining batches carry no labels,
    the probe pulls its own labelled stream from the ImageNet train split and
    runs one step per ``probe_every_n_steps`` optimizer steps. Validation top-1/
    top-5 are evaluated on the ImageNet val split every ``eval_every_n_steps``,
    so the numbers land alongside the zero-shot metrics.

    All probe work happens on rank zero only (the probe parameters are not part
    of the DDP-wrapped module and never trigger collectives), mirroring the rest
    of this file's evaluation callbacks.
    """

    def __init__(
        self,
        cache_dir,
        embed_dim,
        num_heads,
        eval_every_n_steps,
        total_steps,
        batch_size,
        num_workers=8,
        num_classes=1000,
        depth=1,
        lr=1e-3,
        weight_decay=1e-4,
        probe_every_n_steps=1,
    ):
        super().__init__()
        self.cache_dir = cache_dir
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.eval_every = eval_every_n_steps
        self.total_steps = total_steps
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_classes = num_classes
        self.depth = depth
        self.lr = lr
        self.weight_decay = weight_decay
        self.probe_every = max(1, probe_every_n_steps)

        self.classifier = None
        self.optimizer = None
        self.scheduler = None
        self._train_loader = None
        self._train_iter = None
        self._val_loader = None

    def _build(self, pl_module):
        device = pl_module.device
        self.classifier = AttentiveClassifier(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_classes=self.num_classes,
            depth=self.depth,
        ).to(device)
        self.optimizer = torch.optim.AdamW(
            self.classifier.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, self.total_steps // self.probe_every),
            eta_min=self.lr * 0.01,
        )

        loader_kwargs = dict(
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        if self.num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["multiprocessing_context"] = "spawn"
        self._train_loader = DataLoader(
            ImageNetTrain(cache_dir=self.cache_dir), **loader_kwargs
        )
        self._train_iter = iter(self._train_loader)
        self._val_loader = DataLoader(
            ImageNetVal(cache_dir=self.cache_dir),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        print("[probe] online attentive probe initialised", flush=True)

    def _next_batch(self):
        try:
            return next(self._train_iter)
        except StopIteration:
            self._train_iter = iter(self._train_loader)
            return next(self._train_iter)

    @staticmethod
    def _patch_tokens(pl_module, images):
        """Frozen, detached patch tokens (CLS dropped) from the vision encoder."""
        was_training = pl_module.vision_encoder.training
        pl_module.vision_encoder.eval()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            tokens = pl_module.vision_encoder.forward_features(images)
        if was_training:
            pl_module.vision_encoder.train()
        return tokens[:, 1:].float().detach()

    def _train_step(self, trainer, pl_module, step):
        images, labels = self._next_batch()
        images = images.to(pl_module.device, non_blocking=True)
        labels = labels.to(pl_module.device, non_blocking=True)

        tokens = self._patch_tokens(pl_module, images)
        self.classifier.train()
        preds = self.classifier(tokens)
        loss = F.cross_entropy(preds, labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        with torch.no_grad():
            acc = (preds.argmax(dim=-1) == labels).float().mean().item() * 100
        metrics = {
            "train/attentive_probe_loss": loss.item(),
            "train/attentive_probe_acc": acc,
            "train/attentive_probe_lr": self.scheduler.get_last_lr()[0],
        }
        if trainer.logger is not None:
            trainer.logger.log_metrics(metrics, step=step)

    @torch.no_grad()
    def _evaluate(self, trainer, pl_module, step):
        self.classifier.eval()
        top1 = top5 = total = 0
        for images, labels in self._val_loader:
            images = images.to(pl_module.device, non_blocking=True)
            labels = labels.to(pl_module.device, non_blocking=True)
            tokens = self._patch_tokens(pl_module, images)
            logits = self.classifier(tokens)
            top1 += (logits.argmax(dim=-1) == labels).sum().item()
            top5 += (
                (logits.topk(5, dim=-1).indices == labels.unsqueeze(1))
                .any(dim=1)
                .sum()
                .item()
            )
            total += labels.size(0)

        metrics = {
            "eval/attentive_probe_top1": 100 * top1 / total,
            "eval/attentive_probe_top5": 100 * top5 / total,
        }
        print(f"[probe] step={step} metrics={metrics}", flush=True)
        if trainer.logger is not None:
            trainer.logger.log_metrics(metrics, step=step)
        if wandb.run is not None:
            wandb.log({"trainer/global_step": step, **metrics}, commit=True)
            wandb.run.summary.update(metrics)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if _is_global_zero(trainer):
            try:
                if self.classifier is None:
                    self._build(pl_module)
                if step % self.probe_every == 0:
                    self._train_step(trainer, pl_module, step)
                if step > 0 and step % self.eval_every == 0:
                    self._evaluate(trainer, pl_module, step)
            except Exception as e:  # probe must never crash training
                print(f"[probe] failed with: {e}", flush=True)
        if step > 0 and step % self.eval_every == 0:
            trainer.strategy.barrier()


class CheckpointSync(Callback):
    """Saves split text/vision checkpoints and syncs them to a HuggingFace bucket.

    Preserves the original on-disk format (``{run}_text_step{N}.pt`` /
    ``{run}_vision_step{N}.pt``, each a dict of encoder/pre_proj/projector
    state dicts) so existing downstream tooling keeps working.
    """

    def __init__(self, output_dir, run_name, hf_bucket, every_n_steps):
        super().__init__()
        self.output_dir = output_dir
        self.run_name = run_name
        self.hf_bucket = hf_bucket
        self.every = every_n_steps

    def on_train_start(self, trainer, pl_module):
        if _is_global_zero(trainer):
            os.makedirs(self.output_dir, exist_ok=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step == 0 or step % self.every != 0:
            return
        if _is_global_zero(trainer):
            text_ckpt = {
                "encoder": pl_module.text_encoder.state_dict(),
                "pre_proj": pl_module.text_pre_proj.state_dict(),
                "projector": pl_module.projector_text.state_dict(),
            }
            vision_ckpt = {
                "encoder": pl_module.vision_encoder.state_dict(),
                "pre_proj": pl_module.vision_pre_proj.state_dict(),
                "projector": pl_module.projector_vision.state_dict(),
            }
            torch.save(
                text_ckpt, f"{self.output_dir}/{self.run_name}_text_step{step}.pt"
            )
            torch.save(
                vision_ckpt, f"{self.output_dir}/{self.run_name}_vision_step{step}.pt"
            )
            sync_bucket(
                self.output_dir,
                f"hf://buckets/{self.hf_bucket}/{self.run_name}",
                include=[f"{self.run_name}_*.pt"],
                delete=True,
            )
        trainer.strategy.barrier()
