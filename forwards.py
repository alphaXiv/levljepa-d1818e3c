"""Forward function for LeVLJEPA under stable-pretraining.

``levljepa_forward`` is bound to an ``spt.Module`` and called as ``(batch,
stage)``. It returns a dict containing ``loss`` plus the linear/projected
embeddings the ``TrainingMetrics`` callback consumes.
"""

import torch.nn.functional as F

from utils.text import last_unmasked_token


def _cross_align(pred, target, kind):
    """Cross-modal alignment term between a prediction and a (detached) target.

    ``mse``    -- squared Euclidean distance (the original objective).
    ``cosine`` -- BYOL-style normalized prediction on the unit hypersphere,
                  ``2 - 2*cos(pred, target)`` == ``||p_hat - t_hat||^2``. This is
                  exactly the metric zero-shot retrieval uses. Only well-behaved
                  when the target is zero-mean / isotropic (e.g. a SIGReg'd space).
    """
    target = target.detach()
    if kind == "cosine":
        p = F.normalize(pred.float(), dim=-1)
        t = F.normalize(target.float(), dim=-1)
        return (2.0 - 2.0 * (p * t).sum(-1)).mean()
    return (target - pred).square().mean()


def _encode_text_readout(self, batch):
    hidden = self.text_encoder(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    ).last_hidden_state
    text_readout = getattr(self, "text_readout", "eot")
    if text_readout == "eot":
        return last_unmasked_token(hidden, batch["attention_mask"])
    if text_readout == "pad77":
        return hidden[:, -1, :]
    raise ValueError("text_readout must be either 'eot' or 'pad77'.")


def levljepa_forward(self, batch, stage):
    """Single-view cross-modal prediction + per-modality SIGReg (LeVLJEPA).

    Image embeddings predict the stop-gradient text embedding and text
    embeddings predict the stop-gradient image embedding, each through a
    modality-specific MLP predictor. Collapse is prevented by per-modality
    SIGReg, which regularizes each marginal toward an isotropic Gaussian. No
    negatives, no temperature, no momentum encoder.
    """
    text_raw = _encode_text_readout(self, batch)
    image_raw = self.vision_encoder(batch["image"])

    image_linear = self.vision_pre_proj(image_raw)
    text_linear = self.text_pre_proj(text_raw)

    image_proj = self.projector_vision(image_linear)
    text_proj = self.projector_text(text_linear)

    kind = getattr(self, "align_loss", "mse")
    mse_cross_text = _cross_align(image_proj, text_linear, kind)
    mse_cross_vision = _cross_align(text_proj, image_linear, kind)
    mse_alignment = (mse_cross_text + mse_cross_vision) / 2

    sigreg_vision = self.sigreg(image_linear)
    sigreg_text = self.sigreg(text_linear)

    lv, lt = self.lambda_vision, self.lambda_text
    loss = (1 - (lv + lt)) * mse_alignment + lv * sigreg_vision + lt * sigreg_text

    self.log_dict(
        {
            "loss": loss.detach(),
            "sigreg_vision": sigreg_vision.detach(),
            "sigreg_text": sigreg_text.detach(),
            "mse_loss_cross_text": mse_cross_text.detach(),
            "mse_loss_cross_vision": mse_cross_vision.detach(),
            "mse_loss_alignment": mse_alignment.detach(),
        },
        on_step=True,
        on_epoch=False,
        sync_dist=True,
    )

    return {
        "loss": loss,
        "image_linear": image_linear.detach(),
        "text_linear": text_linear.detach(),
        "image_proj": image_proj.detach(),
        "text_proj": text_proj.detach(),
    }
