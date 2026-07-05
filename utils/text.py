import torch


def tokenize_with_eos_readout(tokenizer, texts, max_length=77, padding="max_length"):
    """Tokenize text so EOS is the final unmasked readout token."""
    if max_length < 1:
        raise ValueError("max_length must be at least 1 to hold the EOS readout token.")
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("tokenizer must define eos_token_id for EOS readout pooling.")
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

    single = isinstance(texts, str)
    text_list = [texts] if single else list(texts)
    content_length = max_length - 1
    if content_length > 0:
        encoded = tokenizer(
            text_list,
            add_special_tokens=False,
            padding=False,
            truncation=True,
            max_length=content_length,
            return_attention_mask=False,
        )
        input_ids = encoded["input_ids"]
    else:
        input_ids = [[] for _ in text_list]

    rows = []
    masks = []
    for ids in input_ids:
        row = list(ids[:content_length]) + [eos_id]
        mask = [1] * len(row)
        rows.append(row)
        masks.append(mask)

    if padding == "max_length":
        target_length = max_length
    elif padding is True or padding == "longest":
        target_length = max((len(row) for row in rows), default=1)
    elif padding is False or padding is None:
        target_length = None
    else:
        raise ValueError(
            "padding must be 'max_length', True/'longest', False, or None."
        )

    if target_length is not None:
        for row, mask in zip(rows, masks):
            pad_len = target_length - len(row)
            if pad_len < 0:
                raise ValueError("tokenized row is longer than target padding length.")
            row.extend([pad_id] * pad_len)
            mask.extend([0] * pad_len)

    if target_length is None and len({len(row) for row in rows}) != 1:
        raise ValueError("Cannot return a tensor for unpadded variable-length text.")

    batch = {
        "input_ids": torch.tensor(rows, dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.long),
    }
    return {k: v[:1] for k, v in batch.items()} if single else batch


def last_unmasked_token(hidden, attention_mask):
    """Gather the hidden state at the last attention-mask position."""
    lengths = attention_mask.to(hidden.device).sum(dim=1).clamp(min=1).long()
    gather_idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, hidden.size(-1))
    return hidden.gather(1, gather_idx).squeeze(1)
