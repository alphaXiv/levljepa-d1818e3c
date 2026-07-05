import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as T
from tqdm import tqdm

from utils.text import last_unmasked_token, tokenize_with_eos_readout


PROMPT_TEMPLATE = "a photo of a {}"


class ImageNetVal(Dataset):
    def __init__(self, cache_dir, split="validation"):
        self.ds = load_dataset("ILSVRC/imagenet-1k", split=split, cache_dir=cache_dir)
        self.transform = T.Compose(
            [
                T.ToImage(),
                T.Resize(256),
                T.CenterCrop(224),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        return self.transform(sample["image"].convert("RGB")), sample["label"]

    @property
    def class_names(self):
        return self.ds.features["label"].names


class ImageNetTrain(Dataset):
    """ImageNet train split with standard SSL-probe augmentation.

    Used by the online attentive probe to train a classification head on top of
    frozen backbone features while pretraining runs on CC12M.
    """

    def __init__(self, cache_dir, split="train"):
        self.ds = load_dataset("ILSVRC/imagenet-1k", split=split, cache_dir=cache_dir)
        self.transform = T.Compose(
            [
                T.ToImage(),
                T.RandomResizedCrop(224, scale=(0.08, 1.0), antialias=True),
                T.RandomHorizontalFlip(),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        return self.transform(sample["image"].convert("RGB")), sample["label"]


def _effective_rank(features: torch.Tensor) -> float:
    centered = features - features.mean(0)
    sv = torch.linalg.svdvals(centered.float())
    sv_norm = sv / sv.sum()
    return torch.exp(-torch.sum(sv_norm * torch.log(sv_norm + 1e-7))).item()


def _encode_texts(
    model,
    tokenizer,
    texts,
    device,
    batch_size=64,
    return_full=False,
    text_readout="eot",
    max_length=77,
):
    all_features = []
    all_masks = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        if text_readout == "eot":
            inputs = tokenize_with_eos_readout(
                tokenizer, batch, max_length=max_length, padding="max_length"
            )
        elif text_readout == "pad77":
            inputs = tokenizer(
                batch,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
        else:
            raise ValueError("text_readout must be either 'eot' or 'pad77'.")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            hidden = model(**inputs).last_hidden_state
            if return_full:
                all_features.append(hidden)
                all_masks.append(inputs["attention_mask"] == 0)
            else:
                if text_readout == "eot":
                    all_features.append(
                        last_unmasked_token(hidden, inputs["attention_mask"])
                    )
                else:
                    all_features.append(hidden[:, -1, :])
    if return_full:
        max_len = max(f.size(1) for f in all_features)
        padded_feats = []
        padded_masks = []
        for f, m in zip(all_features, all_masks):
            pad_len = max_len - f.size(1)
            if pad_len > 0:
                f = F.pad(f, (0, 0, 0, pad_len))
                m = F.pad(m, (0, pad_len), value=True)
            padded_feats.append(f)
            padded_masks.append(m)
        return torch.cat(padded_feats, dim=0), torch.cat(padded_masks, dim=0)
    return torch.cat(all_features, dim=0)


def run_imagenet_eval(
    vision_model,
    text_model,
    tokenizer,
    cache_dir,
    device,
    vision_embed=None,
    text_embed=None,
    proj_vision=None,
    proj_text=None,
    batch_size=256,
    num_workers=8,
    proj_cross_eval=True,
    text_readout="eot",
    max_text_length=77,
):
    """Zero-shot ImageNet eval for predictive LeVLJEPA.

    Reports several alignment readouts over the frozen encoders / heads:
      - zeroshot       : image pre_proj  vs text pre_proj
      - proj           : image predictor vs text pre_proj   (image->text prediction)
      - proj_text      : image pre_proj  vs text predictor  (text->image prediction)
      - avg            : mean of proj + proj_text           (the predictive readout)
      - predictor_pair : image predictor vs text predictor

    All models should be unwrapped (no DDP); they are set to eval mode internally
    and restored to their original state on return. Returns a flat dict of
    metrics suitable for ``wandb.log()``.
    """
    if vision_embed is None:
        vision_embed = nn.Identity()
    if text_embed is None:
        text_embed = nn.Identity()

    was_training = {
        "vision": vision_model.training,
        "text": text_model.training,
        "vision_embed": vision_embed.training,
        "text_embed": text_embed.training,
    }
    vision_model.eval()
    text_model.eval()
    vision_embed.eval()
    text_embed.eval()
    use_proj = proj_vision is not None
    if use_proj:
        was_training["proj_vision"] = proj_vision.training
        was_training["proj_text"] = proj_text.training
        proj_vision.eval()
        proj_text.eval()

    dataset = ImageNetVal(cache_dir=cache_dir)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    prompts = [
        PROMPT_TEMPLATE.format(name.replace("_", " ")) for name in dataset.class_names
    ]

    metrics = {}

    with torch.no_grad():
        text_features = text_embed(
            _encode_texts(
                text_model,
                tokenizer,
                prompts,
                device,
                text_readout=text_readout,
                max_length=max_text_length,
            )
        )
        text_features_norm = F.normalize(text_features.float(), dim=-1)
        if use_proj:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                text_proj_norm = F.normalize(
                    proj_text(text_features).float(), dim=-1
                )

        counts = {"zeroshot": [0, 0]}
        if use_proj:
            counts["proj"] = [0, 0]
            counts["predictor_pair"] = [0, 0]
            # proj_text / avg need the predictor output to live in the same space
            # as the pre-projection embeddings (true in predictive mode).
            proj_cross = (
                proj_cross_eval
                and text_proj_norm.shape[-1] == text_features_norm.shape[-1]
            )
            if proj_cross:
                counts["proj_text"] = [0, 0]
                counts["avg"] = [0, 0]

        all_vision_feats_for_rank = []
        total = 0

        def _accumulate(bucket, logits, labels):
            bucket[0] += (logits.argmax(dim=-1) == labels).sum().item()
            bucket[1] += (
                (logits.topk(5, dim=-1).indices == labels.unsqueeze(1))
                .any(dim=1)
                .sum()
                .item()
            )

        for images, labels in tqdm(loader, desc="ImageNet eval"):
            images, labels = images.to(device), labels.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                raw_image_features = vision_model(images)
                image_features = vision_embed(raw_image_features)
            image_norm = F.normalize(image_features.float(), dim=-1)
            all_vision_feats_for_rank.append(image_features.float().cpu())

            _accumulate(counts["zeroshot"], image_norm @ text_features_norm.T, labels)

            if use_proj:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    image_proj_norm = F.normalize(
                        proj_vision(image_features).float(), dim=-1
                    )
                # image predictor (text space) vs text pre_proj.
                logits_proj = image_proj_norm @ text_features_norm.T
                _accumulate(counts["proj"], logits_proj, labels)
                # image predictor vs text predictor.
                _accumulate(
                    counts["predictor_pair"],
                    image_proj_norm @ text_proj_norm.T,
                    labels,
                )
                if proj_cross:
                    # image pre_proj vs text predictor, and the symmetric average.
                    logits_pt = image_norm @ text_proj_norm.T
                    _accumulate(counts["proj_text"], logits_pt, labels)
                    _accumulate(counts["avg"], (logits_proj + logits_pt) / 2, labels)

            total += labels.size(0)

    for name, (top1_n, top5_n) in counts.items():
        metrics[f"eval/{name}_top1"] = 100 * top1_n / total
        metrics[f"eval/{name}_top5"] = 100 * top5_n / total

    vision_feats_all = torch.cat(all_vision_feats_for_rank, dim=0)
    metrics["eval/effective_rank_vision"] = _effective_rank(vision_feats_all)
    metrics["eval/effective_rank_text"] = _effective_rank(text_features.float().cpu())

    if was_training["vision"]:
        vision_model.train()
    if was_training["text"]:
        text_model.train()
    if was_training["vision_embed"]:
        vision_embed.train()
    if was_training["text_embed"]:
        text_embed.train()
    if use_proj:
        if was_training["proj_vision"]:
            proj_vision.train()
        if was_training["proj_text"]:
            proj_text.train()

    return metrics
