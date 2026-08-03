"""Encode the stimuli with a text encoder used by a video generation model.

Only the text tower is loaded, so this runs in minutes and needs no video
generation. Which encoder matters:

  google/umt5-xxl        Wan 2.1 / 2.2                (~11 GB in bf16)
  google/t5-v1_1-xxl     many diffusion models        (~11 GB in bf16)
  openai/clip-vit-large-patch14  CLIP text tower, 77 tokens, useful contrast
  t5-base                smoke test, runs on CPU

Caveat worth keeping in mind when reading the results: diffusion models condition
on the full token sequence through cross-attention, not on a pooled vector. Mean
pooling is a proxy. Pass --save-tokens to keep the per-token states if you want
to redo the analysis without pooling.

Usage:
    python encode.py --model t5-base --out emb_t5base.npz
    python encode.py --model google/umt5-xxl --dtype bfloat16 --device cuda \
        --layers all --out emb_umt5xxl.npz
"""

import argparse
import json
from pathlib import Path

import numpy as np


def load_stimuli(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def flatten(records):
    """-> (texts, item_ids, conditions) in a fixed, reproducible order."""
    texts, item_ids, conditions = [], [], []
    for rec in records:
        for cond, text in rec["texts"].items():
            texts.append(text)
            item_ids.append(rec["item_id"])
            conditions.append(cond)
    return texts, np.array(item_ids), np.array(conditions)


def build_encoder(model_name, device, dtype):
    import torch
    from transformers import AutoTokenizer

    torch_dtype = getattr(torch, dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if "clip" in model_name.lower():
        from transformers import CLIPTextModel

        model = CLIPTextModel.from_pretrained(model_name, torch_dtype=torch_dtype)
    else:
        from transformers import T5EncoderModel

        model = T5EncoderModel.from_pretrained(model_name, torch_dtype=torch_dtype)

    model.eval().to(device)
    return tokenizer, model


def encode(texts, tokenizer, model, device, batch_size, want_layers, save_tokens):
    import torch

    pooled_per_layer, token_states = {}, []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=77, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc, output_hidden_states=want_layers)

            # Mask out padding before pooling; otherwise sentence length leaks
            # into the pooled vector and shows up as a spurious condition effect
            # (the `failed` condition is systematically longer than the rest).
            mask = enc["attention_mask"].unsqueeze(-1).to(out.last_hidden_state.dtype)
            denom = mask.sum(dim=1).clamp(min=1)

            layers = out.hidden_states if want_layers else (out.last_hidden_state,)
            offsets = range(len(layers)) if want_layers else (-1,)
            for layer_idx, hidden in zip(offsets, layers):
                pooled = (hidden * mask).sum(dim=1) / denom
                pooled_per_layer.setdefault(layer_idx, []).append(pooled.float().cpu().numpy())

            if save_tokens:
                token_states.append(
                    (out.last_hidden_state * mask).float().cpu().numpy().astype(np.float16)
                )

    result = {f"layer_{k}": np.concatenate(v, axis=0) for k, v in pooled_per_layer.items()}
    if save_tokens:
        width = max(t.shape[1] for t in token_states)
        padded = [np.pad(t, ((0, 0), (0, width - t.shape[1]), (0, 0))) for t in token_states]
        result["tokens"] = np.concatenate(padded, axis=0)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stimuli", type=Path, default=Path(__file__).parent / "stimuli.jsonl")
    ap.add_argument("--model", default="t5-base")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--layers", choices=["last", "all"], default="last")
    ap.add_argument("--save-tokens", action="store_true")
    args = ap.parse_args()

    records = load_stimuli(args.stimuli)
    texts, item_ids, conditions = flatten(records)
    print(f"encoding {len(texts)} texts with {args.model} on {args.device}")

    tokenizer, model = build_encoder(args.model, args.device, args.dtype)
    arrays = encode(
        texts, tokenizer, model, args.device, args.batch_size,
        want_layers=(args.layers == "all"), save_tokens=args.save_tokens,
    )

    np.savez_compressed(
        args.out,
        item_ids=item_ids,
        conditions=conditions,
        texts=np.array(texts, dtype=object),
        model=np.array(args.model),
        **arrays,
    )
    layer_keys = sorted(k for k in arrays if k.startswith("layer_"))
    print(f"wrote {args.out}  layers={len(layer_keys)}  dim={arrays[layer_keys[-1]].shape[1]}")


if __name__ == "__main__":
    main()
